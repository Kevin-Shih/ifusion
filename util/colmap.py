import argparse
import subprocess
from pathlib import Path
import os
import numpy as np

from PIL import Image
from transforms3d.quaternions import mat2quat

from colmap.database import COLMAPDatabase
from colmap.read_write_model import CAMERA_MODEL_NAMES
from pytorch3d import io
# import open3d as o3d

H, W, NUM_IMAGES = 256, 256, 16

K = np.array([[280., 0., 128.], [0., 280., 128.], [0., 0., 1.]])
POSES = np.array([[[9.99704497e-08, 1.00000000e+00, -4.21468478e-08,
                    -9.82552824e-08], [5.00000000e-01, -9.99704497e-08, -8.66025388e-01, 1.49011612e-08],
                   [-8.66025388e-01, 4.21468478e-08, -5.00000000e-01, 1.49999993e+00]],
                  [[-3.82683516e-01, 9.23879504e-01, -4.47447093e-08,
                    1.08290433e-07], [4.61939782e-01, 1.91341728e-01, -8.66025448e-01, 3.04112255e-08],
                   [-8.00103128e-01, -3.31413627e-01, -5.00000000e-01, 1.49999997e+00]],
                  [[-7.07106709e-01, 7.07106709e-01, 7.93437138e-09,
                    -5.95077854e-09], [3.53553355e-01, 3.53553325e-01, -8.66025329e-01, 7.64635075e-08],
                   [-6.12372339e-01, -6.12372398e-01, -5.00000060e-01, 1.49999985e+00]],
                  [[-9.23879564e-01, 3.82683486e-01, -1.18660761e-08,
                    -4.34263736e-10], [1.91341743e-01, 4.61939752e-01, -8.66025448e-01, 5.87709508e-08],
                   [-3.31413627e-01, -8.00103188e-01, -4.99999970e-01, 1.50000002e+00]],
                  [[-9.99999940e-01, -4.37113883e-08, 6.03368462e-16,
                    -1.80816388e-16], [-2.18556941e-08, 4.99999940e-01, -8.66025388e-01, 9.23298616e-08],
                   [3.78551732e-08, -8.66025388e-01, -5.00000000e-01, 1.49999993e+00]],
                  [[-9.23879623e-01, -3.82683456e-01, 1.99784527e-08,
                    2.96388412e-08], [-1.91341713e-01, 4.61939752e-01, -8.66025448e-01, 2.99236378e-08],
                   [3.31413597e-01, -8.00103188e-01, -4.99999911e-01, 1.50000003e+00]],
                  [[-7.07106709e-01, -7.07106709e-01, -7.93437138e-09,
                    5.95077854e-09], [-3.53553355e-01, 3.53553325e-01, -8.66025329e-01, 7.64635075e-08],
                   [6.12372339e-01, -6.12372398e-01, -5.00000060e-01, 1.49999985e+00]],
                  [[-3.82683516e-01, -9.23879504e-01, 4.47447093e-08,
                    -5.32229230e-08], [-4.61939782e-01, 1.91341728e-01, -8.66025448e-01, 1.90063698e-08],
                   [8.00103128e-01, -3.31413627e-01, -5.00000000e-01, 1.49999999e+00]],
                  [[9.99704497e-08, -1.00000000e+00, 5.64660851e-09,
                    1.20649422e-08], [-4.99999970e-01, -5.78235984e-08, -8.66025388e-01, 5.36155055e-08],
                   [8.66025388e-01, 7.86470906e-08, -4.99999970e-01, 1.49999991e+00]],
                  [[3.82683456e-01, -9.23879504e-01, -4.47447093e-08,
                    3.67556403e-08], [-4.61939782e-01, -1.91341683e-01, -8.66025448e-01, 5.26342383e-08],
                   [8.00103128e-01, 3.31413627e-01, -5.00000000e-01, 1.49999997e+00]],
                  [[7.07106709e-01, -7.07106888e-01, -3.13209902e-09,
                    6.68539579e-09], [-3.53553444e-01, -3.53553355e-01, -8.66025388e-01, 1.16662626e-08],
                   [6.12372518e-01, 6.12372339e-01, -5.00000000e-01, 1.49999992e+00]],
                  [[9.23879445e-01, -3.82683605e-01, -9.62774838e-09,
                    -2.11557556e-08], [-1.91341817e-01, -4.61939752e-01, -8.66025388e-01, 3.58446961e-09],
                   [3.31413716e-01, 8.00103068e-01, -5.00000000e-01, 1.49999989e+00]],
                  [[1.00000000e+00, 1.19248815e-08, 6.13609113e-18,
                    5.09778912e-16], [5.96244032e-09, -5.00000000e-01, -8.66025388e-01, 1.49011611e-08],
                   [-1.03272502e-08, 8.66025388e-01, -5.00000000e-01, 1.49999993e+00]],
                  [[9.23879445e-01, 3.82683605e-01, 9.62774838e-09,
                    2.11557556e-08], [1.91341817e-01, -4.61939752e-01, -8.66025388e-01, 3.58446961e-09],
                   [-3.31413716e-01, 8.00103068e-01, -5.00000000e-01, 1.49999989e+00]],
                  [[7.07106888e-01, 7.07106650e-01, -1.46791166e-08,
                    2.74213363e-09], [3.53553325e-01, -3.53553444e-01, -8.66025448e-01, 6.26714947e-08],
                   [-6.12372339e-01, 6.12372518e-01, -5.00000000e-01, 1.49999995e+00]],
                  [[3.82683367e-01, 9.23879564e-01, -2.07336548e-09,
                    6.21376191e-08], [4.61939812e-01, -1.91341683e-01, -8.66025448e-01, -3.24982481e-08],
                   [-8.00103188e-01, 3.31413567e-01, -5.00000000e-01, 1.50000009e+00]]])

# POSES = POSES[::2]


def extract_and_match_sift(colmap_path, database_path, image_dir, logfile=None):
    cmd = [
        str(colmap_path),
        'feature_extractor',
        '--database_path',
        str(database_path),
        '--image_path',
        str(image_dir),
        '--SiftExtraction.max_image_size',
        '4096',
        '--SiftExtraction.max_num_features',
        '16384',
        '--SiftExtraction.estimate_affine_shape',
        '1',
        '--SiftExtraction.domain_size_pooling',
        '1',
        '--SiftExtraction.peak_threshold',
        '0.0005',
        '--SiftExtraction.edge_threshold',
        '15',
        '--log_level',
        '1',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=logfile, check=True)
    cmd = [
        str(colmap_path),
        'exhaustive_matcher',
        '--database_path',
        str(database_path),
        '--log_level',
        '1',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=logfile, check=True)


def run_triangulation(colmap_path, model_path, in_sparse_model, database_path, image_dir, logfile=None):
    print('Running the triangulation...')
    model_path.mkdir(exist_ok=True, parents=True)
    cmd = [
        str(colmap_path),
        'point_triangulator',
        '--database_path',
        str(database_path),
        '--image_path',
        str(image_dir),
        '--input_path',
        str(in_sparse_model),
        '--output_path',
        str(model_path),
        '--log_level',
        '1',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=logfile, check=False)


def run_patch_match(colmap_path, sparse_model: Path, image_dir: Path, dense_model: Path, logfile=None):
    print('Running patch match...')
    assert sparse_model.exists()
    dense_model.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(colmap_path),
        'image_undistorter',
        '--input_path',
        str(sparse_model),
        '--image_path',
        str(image_dir),
        '--output_path',
        str(dense_model),
        '--log_level',
        '1',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=logfile, check=False)
    cmd = [
        str(colmap_path),
        'patch_match_stereo',
        '--workspace_path',
        str(dense_model),
        '--log_level',
        '1',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=logfile, check=False)


def dump_images(in_image, image_dir, image_name='demo_colmap.png'):
    # print(f'[INFO] in_image shape: {in_image.shape}')               # (256, 4096, 3)
    imgs = np.stack(np.split(in_image, NUM_IMAGES, axis=1), axis=0) # (NUM_IMAGES, 256, 256, 3)
    for index in range(NUM_IMAGES):
        Image.fromarray(imgs[index].astype(np.uint8)).save(f'{str(image_dir)}/{index:03}.png')


def build_db_known_poses_fixed(db_path, in_sparse_path):
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # insert intrinsics
    with open(f'{str(in_sparse_path)}/cameras.txt', 'w') as f:
        for index in range(NUM_IMAGES):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            model, width, height, params = CAMERA_MODEL_NAMES['PINHOLE'].model_id, W, H, np.array((fx, fy, cx, cy),
                                                                                                  np.float32)
            db.add_camera(model, width, height, params, prior_focal_length=(fx + fy) / 2, camera_id=index + 1)
            f.write(f'{index+1} PINHOLE {W} {H} {fx:.3f} {fy:.3f} {cx:.3f} {cy:.3f}\n')

    with open(f'{str(in_sparse_path)}/images.txt', 'w') as f:
        for index in range(NUM_IMAGES):
            pose = POSES[index]
            q = mat2quat(pose[:, :3])
            t = pose[:, 3]
            img_id = db.add_image(f"{index:03}.png", camera_id=index + 1, prior_q=q, prior_t=t)
            f.write(
                f'{img_id} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f} {t[0]:.5f} {t[1]:.5f} {t[2]:.5f} {index+1} {index:03}.png\n\n'
            )

    db.commit()
    db.close()

    with open(f'{in_sparse_path}/points3D.txt', 'w') as f:
        f.write('\n')


def patch_match_with_known_poses(in_image, project_dir, colmap_path='colmap'):
    if os.path.isdir(project_dir):
        # print(f'Confirm Clear project_dir \'{project_dir}\'? [Y/N]')
        # user_input = input()
        # if user_input.lower() in ('y', 'yes'):
        subprocess.run(['rm', '-r', project_dir], check=True)
        # else:
        #     print(f'Canceled: Abort execution.')
        #     exit(0)
    Path(project_dir).mkdir(exist_ok=True, parents=True)

    # output poses
    db_path = f'{str(project_dir)}/database.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    image_dir = Path(f'{str(project_dir)}/images')
    sparse_dir = Path(f'{str(project_dir)}/sparse')
    in_sparse_dir = Path(f'{str(project_dir)}/sparse_in')
    dense_dir = Path(f'{str(project_dir)}/dense')

    image_dir.mkdir(exist_ok=True, parents=True)
    sparse_dir.mkdir(exist_ok=True, parents=True)
    in_sparse_dir.mkdir(exist_ok=True, parents=True)
    dense_dir.mkdir(exist_ok=True, parents=True)
    with open(Path(f'{str(project_dir)}/colmap.txt'), "w") as logfile:
        dump_images(in_image, image_dir)
        build_db_known_poses_fixed(db_path, in_sparse_dir)
        extract_and_match_sift(colmap_path, db_path, image_dir, logfile=logfile)
        run_triangulation(colmap_path, sparse_dir, in_sparse_dir, db_path, image_dir, logfile=logfile)
        run_patch_match(colmap_path, sparse_dir, image_dir, dense_dir, logfile=logfile)

        # fuse
        cmd = [
            str(colmap_path),
            'stereo_fusion',
            '--workspace_path',
            f'{project_dir}/dense',
            '--workspace_format',
            'COLMAP',
            '--input_type',
            'geometric',
            '--output_path',
            f'{project_dir}/points.ply',
            '--log_level',
            '1',
        ]
        print(' '.join(cmd))

        subprocess.run(cmd, stdout=logfile, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--colmap', type=str, default='colmap')
    args = parser.parse_args()
    print(f'Confirm Remove project dir \'{args.project}\'? [Y/N]')
    user_input = input()
    if user_input.lower() in ('y', 'yes'):
        if os.path.isdir(args.project):
            subprocess.run(['rm', '-r', args.project], check=True)
    else:
        print(f'Canceled: Abort execution.')
        exit(0)
    patch_match_with_known_poses(args.dir, args.project, colmap_path=args.colmap)

    vert, _ = io.load_ply(f'{args.project}/points.ply',)
    vn = len(vert)
    with open('colmap-results.log', 'a') as f:
        f.write(f'{args.name}\t{vn}\n')


if __name__ == "__main__":
    # print(K.shape, POSES.shape) # (3, 3) (8, 3, 4)
    main()
