import numpy as np

from carnivalmirror import Sampler
from carnivalmirror import Calibration

class RangeSampler(Sampler):
    """RangeSampler provides ranged-based sampling within the specified range of calibration parameters.

    Attributes:
        width (:obj:`int`): The width of the image(s) for which the calibrations are
        height (:obj:`int`): The height of the image(s) for which the calibrations are
        ranges (:obj:`dict`): The provided sampling ranges for the calibration parameters
        num_of_runs (:obj:`int`): The number of runs for evaluation
    """

    def __init__(self, ranges, cal_width, cal_height, num_of_runs):
        """Initializes a ParameterSampler object
        Args:
            ranges (:obj:`dict`): A dictionary with keys `[fx, fy, cx, cy, k1, k2, p1, p2, k3]` and elements tuples
                describing the sampling range for each parameter. All intrinsic parameters must be provided.
                Missing distortion parameters will be sampled as 0
            cal_width (:obj:`int`): The width of the image(s) for which the calibrations are
            cal_height (:obj:`int`): The height of the image(s) for which the calibrations are
            num_of_runs (:obj:`int`): The number of runs for evaluation
        Raises:
            ValueError: If one of `[fx, fy, cx, cy]` is missing from `ranges`
        """

        super(RangeSampler, self).__init__(cal_width=cal_width, cal_height=cal_height)

        # Validate the ranges
        for key in ['fx', 'fy', 'cx', 'cy']:
            if key not in ranges: raise ValueError("Key %s missing in ranges" % key)
        for key in ['k1', 'k2', 'p1', 'p2', 'k3']:
            if key not in ranges: ranges[key] = (0, 0)
        self.ranges = ranges
        self.num_of_runs = num_of_runs
        self.index = 0


    def next(self):
        """Generator method providing a randomly sampled :obj:`Calibration`
        Returns:
            :obj:`Calibration`: A :obj:`Calibration` object
        """

        # Sample the values
        sample = dict()
        for key in self.ranges:
            sample[key] = self.ranges[key][0] + self.index / self.num_of_runs * (self.ranges[key][1] - self.ranges[key][0])


        # Construct a Calibration object
        K = np.array([[sample['fx'], 0.0, sample['cx']],
                      [0.0, sample['fy'], sample['cy']],
                      [0.0, 0.0, 1.0]])
        D = np.array([sample['k1'], sample['k2'], sample['p1'], sample['p2'], sample['k3']])

        # Check if index is correct
        if self.index > self.num_of_runs:
            raise Exception("Number of times next() is called should not exceed the number of runs!")
        else:
            self.index = self.index + 1

        return Calibration(K=K, D=D, width=self.cal_width, height=self.cal_height)

# EOF
