import math
import numpy as np
import os
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator as rgi
import common
import argparse
import ntpath

# Import shipped libraries.
import librender
import libmcubes
from multiprocessing import Pool
import matplotlib.pyplot as plt

use_gpu = True
if use_gpu:
    import libfusiongpu as libfusion
    from libfusiongpu import tsdf_gpu as compute_tsdf
else:
    import libfusioncpu as libfusion
    from libfusioncpu import tsdf_cpu as compute_tsdf


class Fusion:
    """
    Performs TSDF fusion.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

        self.render_intrinsics = np.array([
            self.options.focal_length_x,
            self.options.focal_length_y,
            self.options.principal_point_x,
            self.options.principal_point_y,
        ], dtype=float)
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrisics = np.array([
            [self.options.focal_length_x, 0, self.options.principal_point_x],
            [0, self.options.focal_length_y, self.options.principal_point_y],
            [0, 0, 1]
        ])
        self.image_size = np.array([
            self.options.image_height,
            self.options.image_width,
        ], dtype=np.int32)
        # Mesh will be centered at (0, 0, 1)!
        self.znf = np.array([
            1 - 0.75,
            1 + 0.75
        ], dtype=float)
        # Derive voxel size from resolution.
        self.voxel_size = 1./self.options.resolution
        self.truncation = self.options.truncation_factor*self.voxel_size

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--mode', type=str, default='render',
                            help='Operation mode: render, fuse or sample.')
        input_group = parser.add_mutually_exclusive_group(required=False)
        input_group.add_argument('--in_dir', type=str,
                                 help='Path to input directory.')
        input_group.add_argument('--in_file', type=str, default='/scratch/liyzhu/MA_Thesis/mesh_processing/watertight.obj', 
                                 help='Path to input directory.')
        parser.add_argument('--out_dir', type=str,
                            help='Path to output directory; files within are overwritten!')
        parser.add_argument('--t_dir', type=str,
                            help='Path to transformation directory.')
        parser.add_argument('--n_proc', type=int, default=0,
                            help='Number of processes to run in parallel'
                                 '(0 means sequential execution).')
        parser.add_argument('--overwrite', action='store_true',
                            help='Overwrites existing files if true.')

        parser.add_argument('--n_points', type=int, default=100000,
                            help='Number of points to sample per model.')
        parser.add_argument('--n_views', type=int, default=12,
                            help='Number of views per model.')
        parser.add_argument('--image_height', type=int, default=640,
                            help='Depth image height.')
        parser.add_argument('--image_width', type=int, default=640,
                            help='Depth image width.')
        parser.add_argument('--focal_length_x', type=float, default=640,
                            help='Focal length in x direction.')
        parser.add_argument('--focal_length_y', type=float, default=640,
                            help='Focal length in y direction.')
        parser.add_argument('--principal_point_x', type=float, default=320,
                            help='Principal point location in x direction.')
        parser.add_argument('--principal_point_y', type=float, default=320,
                            help='Principal point location in y direction.')
        parser.add_argument('--sample_weighted', action='store_true',
                            help='Whether to use weighted sampling.')
        parser.add_argument('--sample_scale', type=float, default=0.2,
                            help='Scale for weighted sampling.')
        parser.add_argument(
            '--depth_offset_factor', type=float, default=1.5,
            help='The depth maps are offsetted using depth_offset_factor*voxel_size.')
        parser.add_argument('--resolution', type=float, default=256,
                            help='Resolution for fusion.')
        parser.add_argument(
            '--truncation_factor', type=float, default=10,
            help='Truncation for fusion is derived as truncation_factor*voxel_size.')

        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def get_in_files(self):
        if self.options.in_dir is not None:
            assert os.path.exists(self.options.in_dir)
            common.makedir(self.options.out_dir)
            files = self.read_directory(self.options.in_dir)
        else:
            files = [self.options.in_file]

        if not self.options.overwrite:
            def file_filter(filepath):
                outpath = self.get_outpath(filepath)
                return not os.path.exists(outpath)
            files = list(filter(file_filter, files))

        return files

    def get_outpath(self, filepath):
        filename = os.path.basename(filepath)
        if self.options.mode == 'render':
            outpath = os.path.join(self.options.out_dir, filename + '.h5')
        elif self.options.mode == 'fuse':
            modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
            # outpath = os.path.join(self.options.out_dir, modelname + '.off')
            outpath = os.path.join(self.options.out_dir, modelname + '.obj')
        elif self.options.mode == 'sample':
            modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
            outpath = os.path.join(self.options.out_dir, modelname + '.npz')

        return outpath
        
    def get_points(self):
        """
        See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

        :param n_points: number of points
        :type n_points: int
        :return: list of points
        :rtype: numpy.ndarray
        """

        rnd = 1.
        points = []
        offset = 2. / self.options.n_views
        increment = math.pi * (3. - math.sqrt(5.))

        for i in range(self.options.n_views):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % self.options.n_views) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        # visualization.plot_point_cloud(np.array(points))
        return np.array(points)

    def get_views(self):
        """
        Generate a set of views to generate depth maps from.

        :param n_views: number of views per axis
        :type n_views: int
        :return: rotation matrices
        :rtype: [numpy.ndarray]
        """

        Rs = []
        points = self.get_points()

        for i in range(points.shape[0]):
            # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
            longitude = - math.atan2(points[i, 0], points[i, 1])
            latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

            R_x = np.array([[1, 0, 0],
                            [0, math.cos(latitude), -math.sin(latitude)],
                            [0, math.sin(latitude), math.cos(latitude)]])
            R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)],
                            [0, 1, 0],
                            [-math.sin(longitude), 0, math.cos(longitude)]])

            R = R_y.dot(R_x)
            Rs.append(R)

        return Rs

    def render(self, mesh, Rs):
        """
        Render the given mesh using the generated views.

        :param base_mesh: mesh to render
        :type base_mesh: mesh.Mesh
        :param Rs: rotation matrices
        :type Rs: [numpy.ndarray]
        :return: depth maps
        :rtype: numpy.ndarray
        """

        depthmaps = []
        for i in range(len(Rs)):
            np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
            np_vertices[2, :] += 1

            np_faces = mesh.faces.astype(np.float64)
            np_faces += 1

            depthmap, mask, img = \
                librender.render(np_vertices.copy(), np_faces.T.copy(),
                                 self.render_intrinsics, self.znf, self.image_size)
            plt.imshow(depthmap)
            plt.show()
            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs).
            depthmap -= self.options.depth_offset_factor * self.voxel_size
            depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))
            

            depthmaps.append(depthmap)

        return depthmaps
    def run_render(self, filepath):
        """
        Run rendering.
        """
        timer = common.Timer()
        Rs = self.get_views()

        timer.reset()
        # mesh = common.Mesh.from_off(filepath)
        mesh = common.Mesh.from_obj(filepath)
        depths = self.render(mesh, Rs)

        depth_file = self.get_outpath(filepath)
        common.write_hdf5(depth_file, np.array(depths))
        print('[Data] wrote %s (%f seconds)' % (depth_file, timer.elapsed()))
    
if __name__ == "__main__":
    render = Fusion()
    # Rs = render.get_views()
    render.run_render('/scratch/liyzhu/MA_Thesis/mesh_processing/watertight.obj')
    pass