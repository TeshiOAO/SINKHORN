import yt
import numpy as np
from absl import flags
from absl.flags import FLAGS
import sys
import os
import shutil

flags.DEFINE_string('timeStep', "1",'time step of out put')
flags.DEFINE_string('inputDir', "inputfile",'path to load file')
# flags.DEFINE_string('size', None,'resolution of simulation output')
flags.DEFINE_string('outputDir', "outputfile",'path to load file')
time = ['00302', '00316', '00332', '00351', '00374', '00403']

def main(argv, file):
        inputfolder, outputfolder, inputfile = file
        outputfolder = os.path.join(outputfolder, inputfile)
        os.mkdir(outputfolder)

        for t in time:
            t = 'plt' + t

            FLAGS.inputDir = os.path.join(inputfolder, inputfile, t)
            FLAGS.outputDir = os.path.join(outputfolder, t)
            os.mkdir(os.path.join(outputfolder, t))

            argv = FLAGS(argv)  # parse flags
            ds = yt.load(FLAGS.inputDir,hint='NyxDataset')
            ad = ds.all_data()
            for i in range(len(ds.field_list)):
                if ds.field_list[i][0] == 'boxlib':
                    collect = -1* np.ones((64, 64, 64))
                    count = 0
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                collect[x*32:(x+1)*32, y*32:(y+1)*32, z*32:(z+1)*32] = ad[('boxlib', ds.field_list[i][1])][(count)*32768:(count+1)*32768].reshape(32, 32, 32, order='F') 
                                count+=1
                    #print(collect.shape)
                    collect.astype(np.float32).tofile(os.path.join(FLAGS.outputDir, f'{ds.field_list[i][1]}{FLAGS.timeStep}.bin'))


            try:
                shutil.rmtree(FLAGS.inputDir)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        try:
            shutil.rmtree(os.path.join(inputfolder, inputfile))
        except OSError as e:
            print(f"Error:{ e.strerror}")

if __name__ == '__main__':
    inputfolder = './inputfiles'
    outputfolder = './outputfiles'
    inputfiles = os.listdir(inputfolder)

    for inputfile in inputfiles:
        main(sys.argv, [inputfolder, outputfolder, inputfile])