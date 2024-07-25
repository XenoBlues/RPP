import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class FR5(DHRobot):
    """
    Class that models a Universal Robotics UR5 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``UR5()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.UR5()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR4`, :func:`UR10`


    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        # robot length values (metres)
        a = [0, -0.425, -0.395, 0, 0, 0]
        d = [0.152, 0, 0, 0.102, 0.102, 0.1]

        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data, no inertia available
        mass = [4.64, 10.08, 2.71, 1.56, 1.56, 0.36]
        center_of_mass = [
            [-0.00019, -0.01828, 0.00226],
            [0.21247, 0, 0.00012102],
            [0.12262, 0.00017, 0.01259],
            [0.00005, -0.00233, 0.01468],
            [-0.00005, 0.00233, 0.01468],
            [0.00093, 0.00081, -0.02005],
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=center_of_mass[j], G=1,
            )
            links.append(link)

        super().__init__(
            links,
            name="FR5",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([0, 90, 90, 90, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    fr5 = FR5(symbolic=False)
    print(fr5)
    # print(ur5.dyntable())
