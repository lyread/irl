import numpy as np

from environments.robot_arms.arm_common import ArmCommon


class NAO4DoF(ArmCommon):
    def __init__(
        self,
        name: str = "NAO4",
        task_space_dims=3,
        goal_zone_radius=17.5,
        max_step_length=np.pi / 10,
        dataset_path=None,
        training_episodes=100,
        test_episodes=100,
        rescale_state=False,
        shuffle_mode="keep_pairings",
        enable_punishment=False,
    ):
        joint_ranges = np.array(
            [[-2.0857, 2.0857], [-1.3265, 0.3142], [-2.0857, 2.0857], [0.0349, 1.5446]]
        )  # in rad according to NAO's documentation
        link_lengths = np.array(
            [
                [0.0, 98.0, 100.0],
                [0.0, 0.0, 0.0],
                [105.0, 15.0, 0.0],
                [0.0, 0.0, 0.0],
                [55.95, 0.0, 0.0],
            ]
        )  # in mm according to NAO's documentation

        super().__init__(
            name=name,
            joint_ranges=joint_ranges,
            link_lengths=link_lengths,
            task_space_dims=task_space_dims,
            goal_zone_radius=goal_zone_radius,
            max_step_length=max_step_length,
            dataset_path=dataset_path,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            rescale_state=rescale_state,
            shuffle_mode=shuffle_mode,
        )

    def end_effector_coordinates(self, angles):
        """
        Calculate the coordinates of the end effector given the joint angles.

        Args:
            angles (np.ndarray): rotation angles.

        Returns:
            np.ndarray: End effector coordinates.
        """
        assert len(angles) == 4, "%s == 4" % len(angles)

        UAL = self.link_lengths[2][0]  # Upper Arm Length
        LAL = self.link_lengths[4][0]  # Lower Arm Length
        elbowOffset = self.link_lengths[2][1]  # Elbow Offset

        corrected_angles = self.validate_angles(angles)

        # transformation matrices
        c = np.cos(corrected_angles)
        s = np.sin(corrected_angles)

        T01 = np.array(
            [
                [c[0], -s[0], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-s[0], -c[0], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T12 = np.array(
            [
                [c[1], -s[1], 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [s[1], c[1], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T23 = np.array(
            [
                [c[2], -s[2], 0.0, -elbowOffset],
                [0.0, 0.0, -1.0, -UAL],
                [s[2], c[2], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T34 = np.array(
            [
                [c[3], -s[3], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-s[3], -c[3], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T24 = np.matmul(T23, T34)
        T14 = np.matmul(T12, T24)
        T04 = np.matmul(T01, T14)  # OK was tested

        R04 = T04[0:3, 0:3]
        O0O4_Base = T04[:3, 3]  # O0O4 in base coordinates
        O4O5_R4 = np.array([LAL, 0.0, 0.0]).T  # O4O5 in end effector coordinates (R4)
        pointInBase = O0O4_Base + np.matmul(
            R04, O4O5_R4
        )  # Xp = O0O5(0) = O0O4(0) + O4O5(0) = O0O4(0) + R04 * O4O5(4)

        return pointInBase
