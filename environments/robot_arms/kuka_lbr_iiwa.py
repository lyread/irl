import numpy as np

from environments.robot_arms.arm_common import ArmCommon


class KUKA_LBR_IIWA(ArmCommon):
    def __init__(
        self,
        name: str = "KUKA7",
        task_space_dims=3,
        goal_zone_radius=150,
        max_step_length=np.pi / 10,
        dataset_path=None,
        training_episodes=100,
        test_episodes=100,
        rescale_state=False,
        shuffle_mode="keep_pairings",
    ):
        joint_ranges = np.array(
            [
                [-2.96706, 2.96706],
                [-2.0944, 2.0944],
                [-2.96706, 2.96706],
                [-2.0944, 2.0944],
                [-2.96706, 2.96706],
                [-2.0944, 2.0944],
                [-3.05433, 3.05433],
            ]
        )

        link_lengths = np.array(
            [
                [0, 0, 0],
                [0, 0, 360],
                [0, 0, 0],
                [0, 0, 420],
                [0, 0, 0],
                [0, 0, 400],
                [0, 0, 0],
                [0, 0, 126],
            ]
        )

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
        Returns the coordinates of O8 in base coordinates (R0).
            Parameters:
                angles -- the angles to use for the transformations, cannot be
                          outside the specified ranges
                returns -- the position of O8 (Origin of the End Effector)
                           in base coordinates (R0)
        """
        assert len(angles) == 7, "%s == 7" % len(angles)
        corrected_angles = self.validate_angles(angles)

        # transformation matrices
        c = np.cos(corrected_angles)
        s = np.sin(corrected_angles)

        T01 = np.array(
            [
                [c[0], -s[0], 0.0, 0.0],
                [s[0], c[0], 0.0, 0.0],
                [0.0, 0.0, 1.0, self.link_lengths[1, 2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T12 = np.array(
            [
                [c[1], -s[1], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-s[1], -c[1], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T23 = np.array(
            [
                [c[2], -s[2], 0.0, 0.0],
                [0.0, 0.0, -1.0, -self.link_lengths[3, 2]],
                [s[2], c[2], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T34 = np.array(
            [
                [c[3], -s[3], 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [s[3], c[3], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T45 = np.array(
            [
                [c[4], -s[4], 0.0, 0.0],
                [0.0, 0.0, 1.0, self.link_lengths[5, 2]],
                [-s[4], -c[4], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T56 = np.array(
            [
                [c[5], -s[5], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-s[5], -c[5], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T67 = np.array(
            [
                [c[6], -s[6], 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [s[6], c[6], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # FIXME: this could be simplified
        # T67 = np.array([[c[6], -s[6], 0.0, 0.0],
        #                     [0.0, 0.0, -1.0, -self.link_lengths[7, 2]],
        #                     [s[6], c[6], 0.0, 0.0],
        #                     [0.0, 0.0, 0.0, 1.0]])
        #

        T57 = np.matmul(T56, T67)
        T47 = np.matmul(T45, T57)
        T37 = np.matmul(T34, T47)
        T27 = np.matmul(T23, T37)
        T17 = np.matmul(T12, T27)
        T07 = np.matmul(T01, T17)  # OK was tested

        # FIXME: Then do this instead of the whole R07 etc business
        # pointInBase = T07[:3, 3]

        R07 = T07[0:3, 0:3]
        O0O7_Base = T07[:3, 3]  # O0O7 in base coordinates
        # FIXME: handle the dtype properly (different when runs on cpu vs gpu)
        # O7O8 in end effector coordinates (R7)
        O7O8_R7 = np.array([0.0, 0.0, self.link_lengths[7, 2]]).T
        # Xp = O0O8(0) = O0O7(0) + O7O8(0) = O0O7(0) + R07 * O7O8(0)
        pointInBase = O0O7_Base + np.matmul(R07, O7O8_R7)

        return pointInBase
