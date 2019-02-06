import argparse
import tensorflow as tf
from open3d import *
import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/pyramid_roof.pcd')
    parser.add_argument('--model_type', default='pcn_cd')
    parser.add_argument('--checkpoint', default='log/model-1850')
    parser.add_argument('--num_gt_points', type=int, default=1000)
    args = parser.parse_args()

    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, tf.constant(1.0))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    partial = read_point_cloud(args.input_path)
    partial = np.array(partial.points)
    complete = sess.run(model.outputs, feed_dict={inputs: [partial]})[0]

    import pptk

    print('Partial Point Cloud')
    v6 = pptk.viewer(partial)
    v6.set(point_size=0.1)

    print('Completed Point Cloud')
    v6 = pptk.viewer(complete)
    v6.set(point_size=0.005)

