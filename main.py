from __future__ import division
import numpy as np
from numpy import *
import argparse
from sklearn.neighbors import NearestNeighbors as nb
from scipy.sparse import *
import sys
import yaml
import logging

log = logging.getLogger('main');
log.setLevel(logging.DEBUG);

#================================================================================
#	Supplementary Code
def save_sparse_csc(filename,array):
	np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )

def update_progress(progress):
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def makedirs(directories):
    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory);
#=================================================================================

def main(config):

    saveDir = config['saveDir'];
    figDir  = config['figDir'];


    directories = [ saveDir,
                    figDir,
                    os.path.join(saveDir, '.memmapped'),
                    os.path.join(saveDir, '.debug'),
                  ];
    makedirs(directories);

    if config['icafile'][-3:] == 'npy':
        icacoffs = np.load(config['icafile']);
        log.warning('Due to the typically large ICA file size, we recommend '+\
                    'running this on the memory mapped array, typically of the '+\
                    'form *icacoffs*.array',
                   );
    elif 'icadim' in config:
        if config['icafile'][-5:] == 'array':
            icacoffs = np.memmap(config['icafile'], dtype='float64').reshape(config['icadim'],-1);
    elif config['icafile'][-5:] == 'array':
        raise IOError('If using memmapped array as input, you must specify the ICA dimension as \'icadim\' in your config.yaml.');
    else: raise ValueError('Please enter valid array or memmapped array for ICA.');

    numSamples = icacoffs.shape[1];
    log.debug('{0}'.format(numSamples));
    numSamples = numSamples - numSamples % config['numBlock'];	#	Just so it fits in nicely (Losing a few samples won't matter.)
    log.debug('new numSamples: {0}'.format(numSamples));

    #	KNN		-----------------------------------------------------------------------------

    #	This step determines how many dimensions we should consider when computing the k-neighbors	
    if config['setup']:
	    pts = [];
	    timing.log('Setup to determine adequate affinity subspace...');
	    for i in range(3,config['icadim']):
		    print('Finding neighbors in %i dimensions...' %(i));
		    #	Slice every 10 to save time
		    nbrs = nb(algorithm='kd_tree').fit(icacoffs[:i,::10].T);
		    ind = nbrs.radius_neighbors(icacoffs[:i,::10].T, return_distance=False);
		    sz = [];
		    for n in ind:
			    sz.append(len(n));
		    pts.append(np.mean(sz));
		    if np.mean(sz) < 10:
			    print 'Average number of neighbors has fallen below 10...'
			    break;
	    plt.plot( np.arange(2,len(pts)+2) , np.array(pts) );
	    plt.yscale('log');
	    plt.xscale('log');
	    plt.xlabel('Number of Dimensions Considered')
	    plt.ylabel('Average Number of Neighbors')
	    plt.title('Average Neighbors within a Unit Radius of a Point') 
        plt.savefig( os.path.join(config['figDir'], '{0}_radneigh.png'.format(config['pname'])) );
        pickle.dump( fig, file( os.path.join(config['figDir'], '{0}_radneigh.png'.format(config['pname'])), 'w+') );
        if 'graph' in config and config['graph']:
    	    plt.show();
        plt.clf();
	    a = input('Enter desired affinity dimension (enter # <= 0 for default): ');
	    if (a > 0):
		    config['affdim'] = a;

	    #	Have to transpose icacoffs b/c sklearn wants Samples x Sensors

    subset = np.memmap(os.path.join(saveDir, '.memmapped/subset.array'), dtype='float64', mode='w+', shape=(config['affdim'], numSamples));
    subset[:,:] = icacoffs[ :config['affdim'], :numSamples ];
    blockwidth = numSamples//config['numBlock'];

    tmp_ind = np.memmap(os.path.join(saveDir, '.memmapped/tmpind.array'), mode='w+', dtype='int64', shape=(blockwidth,config['numBlock']*config['n_neighbors']));
    tmp_dist = np.memmap(os.path.join(saveDir, '.memmapped/tmpdist.array'), mode='w+', dtype='float64', shape=(blockwidth,config['numBlock']*config['n_neighbors']));

    distances = np.memmap(os.path.join(saveDir, '.memmapped/distances.array'), mode='w+', dtype='float64', shape=(numSamples,config['n_neighbors']) );
    indices = np.memmap(os.path.join(saveDir, '.memmapped/indices.array'), mode='w+', dtype='int64', shape=(numSamples,config['n_neighbors']) );

    log.info('Beginning search for {0} nearest neighbors using only {1} dimensions...'.format(config['n_neighbors'], config['affdim']));

    for i in range(config['numBlock']):

	    for j in range(config['numBlock']):
		    #	Would like to use symmetry to our advantage at some point, but I haven't implemented this yet.
		    #if (i <= j):
			    nbrs = nb(n_neighbors=config['n_neighbors'], algorithm='kd_tree').fit(subset[:,j*blockwidth:(j+1)*blockwidth].T);
			    dist, ind = nbrs.kneighbors(subset[:,i*blockwidth:(i+1)*blockwidth].T);
			    ind += blockwidth*j;
			    tmp_ind[:, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = ind;
			    tmp_dist[:, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = dist;
			    del ind, dist, nbrs;
			    update_progress( (j+i*config['numBlock']+1) / float(config['numBlock']**2) ); # +1 gets us to 100%
	    indsort = np.argsort(tmp_dist,axis=1);
	
	    for j in range(blockwidth):
		    tmp_ind[j,:] = tmp_ind[j,indsort[j]];
		    tmp_dist[j,:] = tmp_dist[j,indsort[j]];

	    indices[i*blockwidth:(i+1)*blockwidth,:] = tmp_ind[:,:config['n_neighbors']];
	    distances[i*blockwidth:(i+1)*blockwidth,:] = tmp_dist[:,:config['n_neighbors']];
	    indices.flush(); distances.flush();
			
		    #"""else:
		    #	indices[i*blockwidth:(i+1)*blockwidth, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = \
		    #		indices[i*blockwidth:(i+1)*blockwidth, j*config['n_neighbors']:(j+1)*config['n_neighbors']]
		    #"""

    distances = np.memmap(os.path.join(saveDir, '.memmapped/distances.array'), dtype='float64');
    indices = np.memmap(os.path.join(saveDir, '.memmapped/indices.array'), dtype='int64');
    affdata = np.memmap(os.path.join(saveDir, '.memmapped/data.array'), 
                        mode='w+',
                        dtype='float64',
                        shape=(numSamples*config['n_neighbors']),
                       );
    affdata[:] = np.exp(-distances);
		

    np.save(os.path.join(saveDir, '.debug','indices_%s_%in.npy' %(config['pname'], config['n_neighbors']), indices[:]);

    log.info('Search complete...\n\nBeginning affinity matrix generation...');

    #	Affgen	-----------------------------------------------------------------------------

    indptr = np.arange(0,config['n_neighbors']*numSamples+1, config['n_neighbors']);
    affMat = csc_matrix( (affdata[:], indices[:], indptr), shape = (numSamples,numSamples) );

    #   Ugly, but more readable than a long list...
    save_sparse_csc(os.path.join(saveDir, 
                                '{0}_aff_{1}d_{2}n.npz'.format(config['pname'], 
                                                               config['icadim'],
                                                               config['n_neighbors'],
                                                              ),
                                ),
                    affMat,
                   );

    if 'graph' in config and config['graph']:
    	plt.spy(affMat, precision = .1, markersize = 1);
    	plt.show();

    log.info('Affinity generation complete...');

if __name__ == '__main__':

    parser = argparse.ArgumentParser();
    parser.add_argument('-g', action='store_true', dest='graph', default=False, 
                        help='Shows graph of Affinity Matrix and eigenv\'s, depending on flags.');
    parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.');
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.');
    parser.add_argument('--setup', action='store_true', dest='setup', default=False, 
                        help='Runs setup calculations: Unit radius neighbor search.');

    values = parser.parse_args()

    #   Get config from file
    with open(values.configpath) as f:
        conf_file = f.read();
        config = yaml.load(conf_file);
    if not 'config' in locals(): raise IOError(
    'Issue opening and reading configuration file: {0}'.format(os.path.abspath(values.configpath)) );

    #   Update config with CLARGS
    level = 30;
    if values.verbose: level = 20;
    elif values.debug: level = 10;
    config['graph'] = values.graph;
    config['setup'] = values.setup;

    #   Setup stream logger
    ch = logging.StreamHandler(sys.stdout);
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s');
    ch.setLevel(level);
    ch.setFormatter(formatter);

    log.addHandler(ch);

    log.debug('Configuration File:\n'+conf_file);
    log.info('Using Configuration File: {0}'.format(os.path.abspath(values.configpath)));

    if os.path.isfile( config['icafile'] ):
        log.info('Generating affinity of supplied ICA file: {0}'.format(config['icafile']));
        main(config);
    else:
        raise IOError('Please input a valid ICA file.  Your path: ({0}) was invalid.'.format(config['icafile']));
