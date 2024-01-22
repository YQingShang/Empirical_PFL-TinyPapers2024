import os
from multiprocessing import Pool

def run_local_epoch(local_epoch):
    output_filename = '~/*/local_epoch{}_{}.txt'
    os.system('echo "A new global round" > ' + output_filename)
    base_command = ('python main.py -datp ~/* -data Continuous_npz -m lr -algo PerAvg -gr 10 -did 0 -nb 2 -nc 2 -le {} -lr 1 -lbs 128 >> '
                    + output_filename)
    echo_command = 'echo "A new global round" >> ' + output_filename
    le_value = f"{local_epoch[0]},{local_epoch[1]}"
    command = base_command.format(le_value, local_epoch[0], local_epoch[1])
    os.system(command)
    os.system(echo_command)

if __name__ == "__main__":
    le = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), 
          (10, 10), (10, 12), (10, 15), (10, 20), (10, 25), (10, 35), (10, 50), (10, 70)]
    with Pool(len(le)) as pool:
        pool.map(run_local_epoch, le)

(60,5)
