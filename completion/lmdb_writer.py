# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow


class pcd_df(DataFlow):
    def __init__(self, model_list, num_scans, partial_dir, complete_dir):
        self.model_list = model_list
        self.num_scans = num_scans
        self.partial_dir = partial_dir
        self.complete_dir = complete_dir

    def size(self):
        return len(self.model_list) * self.num_scans

    def get_data(self):
        for model_id in model_list:
            complete = read_pcd(os.path.join(self.complete_dir, '%s.pcd' % model_id))
            for i in range(self.num_scans):
                read_path = os.path.join(self.partial_dir, model_id, '%d.pcd' % i)
                if os.path.exists(read_path):
                    partial = read_pcd(read_path)
                    yield model_id.replace('/', '_'), partial, complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/data/damaged_v10_1000pts_meansubval/val.list')
    parser.add_argument('--num_scans', default=8, type=int)
    parser.add_argument('--partial_dir', default='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/data/damaged_v10_1000pts_meansubval/partial')
    parser.add_argument('--complete_dir', default='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/data/damaged_v10_1000pts_meansubval/complete')
    parser.add_argument('--output_path', default='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/data/damaged_v10_1000pts_meansubval')
    args = parser.parse_args()

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    df = pcd_df(model_list, args.num_scans, args.partial_dir, args.complete_dir)
    if os.path.exists(args.output_path):
        os.system('rm %s' % args.output_path)
    dataflow.LMDBSerializer.save(df, args.output_path)
