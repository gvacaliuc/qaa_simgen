from __future__ import division
import numpy as np
from numpy import *
import argparse
from sklearn.neighbors import NearestNeighbors as nb
from scipy.sparse import *
import sys
import yaml
import logging
import os
import matplotlib.pyplot as plt
import pickle

log = logging.getLogger('main');
log.setLevel(logging.DEBUG);

#================================================================================
#    Supplementary Code
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
    numSamples = numSamples - numSamples % config['numblock'];    #    Just so it fits in nicely (Losing a few samples won't matter.)
    log.debug('new numSamples: {0}'.format(numSamples));

    #    KNN        -----------------------------------------------------------------------------

    #    This step determines how many dimensions we should consider when computing the k-neighbors
    pts = [];
    log.info('Setup to determine adequate affinity subspace...');
    for i in range(3,icacoffs.shape[0]):
        log.info('Finding neighbors in %i dimensions...' %(i));
        #    Slice every 10 to save time
        nbrs = nb(algorithm='kd_tree').fit(icacoffs[:i,::10].T);
        ind = nbrs.radius_neighbors(icacoffs[:i,::10].T, return_distance=False);
        sz = [];
        for n in ind:
            sz.append(len(n));
        pts.append(np.mean(sz));
        if np.mean(sz) < 10:
            log.info('Average number of neighbors has fallen below 10...');
            break;
    fig = plt.figure();
    ax = fig.gca();
    ax.plot( np.arange(2,len(pts)+2) , np.array(pts) );
    ax.set_yscale('log');
    ax.set_xscale('log');
    ax.set_xlabel('Number of Dimensions Considered')
    ax.set_ylabel('Average Number of Neighbors')
    ax.set_title('Average Neighbors within a Unit Radius of a Point') 
    plt.savefig( os.path.join(config['figDir'], '{0}_radneigh.png'.format(config['pname'])) );
    pickle.dump( fig, file( os.path.join(config['figDir'], '{0}_radneigh.pickle'.format(config['pname'])), 'w+') );
    if ('graph' in config and config['graph']) or ('setup' in config and config['setup']):
        plt.show();
    plt.clf();
    if 'setup' in config and config['setup']:
        a = input('Enter desired affinity dimension (enter # <= 0 for default): ');
        if (a > 0):
            config['affdim'] = a;

        #    Have to transpose icacoffs b/c sklearn wants Samples x Sensors

    subset = np.memmap(
        os.path.join(
            saveDir,
            '.memmapped/subset.array',
                    ),
        dtype='float64',
        mode='w+',
        shape=(config['affdim'], numSamples),
                      );

    subset[:,:] = icacoffs[ :config['affdim'], :numSamples ];
    blockwidth = numSamples//config['numblock'];

    tmp_ind = np.memmap(
        os.path.join(
            saveDir,
            '.memmapped/tmpind.array',
                    ),
        dtype='int64',
        mode='w+',
        shape=(blockwidth,config['numblock']*config['n_neighbors']),
                      );
    tmp_dist = np.memmap(
        os.path.join(
            saveDir,
            '.memmapped/tmpdist.array',
                    ),
        dtype='float64',
        mode='w+',
        shape=(blockwidth,config['numblock']*config['n_neighbors']),
                      );
    distances = np.memmap(
        os.path.join(
            saveDir,
            '.memmapped/distances.array',
                    ),
        dtype='float64',
        mode='w+',
        shape=(numSamples,config['n_neighbors']),
                      );
    indices = np.memmap(
        os.path.join(
            saveDir,
            '.memmapped/indices.array',
                    ),
        dtype='int64',
        mode='w+',
        shape=(numSamples,config['n_neighbors']),
                      );

    log.info('Beginning search for {0} nearest neighbors using only {1} dimensions...'.format(config['n_neighbors'], config['affdim']));

    for i in range(config['numblock']):

        for j in range(config['numblock']):
            #    Would like to exploit symmetry but I haven't implemented this yet.
            #if (i <= j):
                nbrs = nb(n_neighbors=config['n_neighbors'], algorithm='kd_tree').fit(subset[:,j*blockwidth:(j+1)*blockwidth].T);
                dist, ind = nbrs.kneighbors(subset[:,i*blockwidth:(i+1)*blockwidth].T);
                ind += blockwidth*j;
                tmp_ind[:, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = ind;
                tmp_dist[:, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = dist;
                del ind, dist, nbrs;
                update_progress( (j+i*config['numblock']+1) / float(config['numblock']**2) ); # +1 gets us to 100%
        indsort = np.argsort(tmp_dist,axis=1);
    
        for j in range(blockwidth):
            tmp_ind[j,:] = tmp_ind[j,indsort[j]];
            tmp_dist[j,:] = tmp_dist[j,indsort[j]];

        indices[i*blockwidth:(i+1)*blockwidth,:] = tmp_ind[:,:config['n_neighbors']];
        distances[i*blockwidth:(i+1)*blockwidth,:] = tmp_dist[:,:config['n_neighbors']];
        indices.flush(); distances.flush();
            
            #"""else:
            #    indices[i*blockwidth:(i+1)*blockwidth, j*config['n_neighbors']:(j+1)*config['n_neighbors']] = \
            #        indices[i*blockwidth:(i+1)*blockwidth, j*config['n_neighbors']:(j+1)*config['n_neighbors']]
            #"""

    distances = np.memmap(os.path.join(saveDir, '.memmapped/distances.array'), dtype='float64');
    indices = np.memmap(os.path.join(saveDir, '.memmapped/indices.array'), dtype='int64');
    affdata = np.memmap(os.path.join(saveDir, '.memmapped/data.array'), 
                        mode='w+',
                        dtype='float64',
                        shape=(numSamples*config['n_neighbors']),
                       );
    affdata[:] = np.exp(-distances);
        

    np.save(os.path.join(saveDir, '.debug','indices_%s_%in.npy' %(config['pname'], config['n_neighbors'])), indices[:]);

    log.info('Search complete...\n\nBeginning affinity matrix generation...');

    #    Affgen    -----------------------------------------------------------------------------

    indptr = np.arange(0,config['n_neighbors']*numSamples+1, config['n_neighbors']);
    affMat = csc_matrix( (affdata[:], indices[:], indptr), shape = (numSamples,numSamples) );

    #   Easy way to score symmetry
    affMat = affMat + affMat.T;

    #   Ugly, but more readable than a long list...
    save_sparse_csc(os.path.join(saveDir, 
                                '{0}_aff_{1}d_{2}n.npz'.format(config['pname'], 
                                                               icacoffs.shape[0],
                                                               config['n_neighbors'],
                                                              ),
                                ),
                    affMat,
                   );

    #   Makes figure
    fig = plt.figure();
    ax = fig.gca();
    ax.set_title('{0} Affinity Matrix, subsampled data to {1} dimensions'.format(config['pname'], config['affdim']));
    aff_fig = ax.spy(affMat, precision = .01, markersize = 1);
    #fig.colorbar(aff_fig);
    
    filename = os.path.join(figDir, 
                                '{0}_aff_{1}d_{2}n'.format(config['pname'], 
                                                               icacoffs.shape[0],
                                                               config['n_neighbors'],
                                                              ),
                            );
    plt.savefig(filename+'.png');
    pickle.dump(fig, file(filename+'.pickle', 'w+') );

    if 'graph' in config and config['graph']:
        plt.show();

    log.info('Affinity generation complete...');

#================================================
#   Validate:
#   
#   Validates our config file, making directories if necessary
#================================================
def validate(config):
    
    required = ['numblock', 'logfile', 'saveDir', 'figDir','pname', 'n_neighbors', 'affdim','icafile',];
    for field in required:
        if not field in config:
            raise ValueError('You didn\'t provide a value for the field: {0}'.format(field));

    directories = ['saveDir', 'figDir',];
    for directory in directories:
        if not os.path.isdir(config[directory]):
            os.makedirs(config[directory]);

    if config['icafile'][-5:] == 'array' and not 'icadim' in config:
        raise ValueError('We need the first dimension if you\'re passing a memory mapped array.');

    if not os.path.isdir(os.path.join(config['saveDir'], '.memmapped')):
        os.makedirs(os.path.join(config['saveDir'], '.memmapped'));
    if not os.path.isdir(os.path.join(config['saveDir'], '.debug',)):
        os.makedirs(os.path.join(config['saveDir'], '.debug',));


if __name__ == '__main__':

    parser = argparse.ArgumentParser();
    parser.add_argument('-g', action='store_true', dest='graph', default=False, 
                        help='Shows graph of Affinity Matrix and eigenv\'s, depending on flags.');
    parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.');
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.');
    parser.add_argument('--setup', action='store_true', dest='setup', default=False, 
                        help='Runs setup calculations: Unit radius neighbor search.');
    parser.add_argument('--config', type=str, dest='configpath', default='config.yaml',
                        help='Input other configuration file.');

    values = parser.parse_args()

    #   Get config from file
    with open(values.configpath) as f:
        conf_file = f.read();
        config = yaml.load(conf_file);
    if not 'config' in locals(): raise IOError(
    'Issue opening and reading configuration file: {0}'.format(os.path.abspath(values.configpath)) );

    validate(config);

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
