import numpy as np
import torch

from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from Box2D import (
    b2World,
    b2PolygonShape,
    b2CircleShape,
    b2_staticBody,
    b2_dynamicBody,
    b2Vec2,
)


from ..base import BenchmarkProblem, DataType


class RobotPush(BenchmarkProblem):
    """
    https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/4e6f9ed04833cc2e21b5906b1181bc067298f914
    """

    available_dimensions = 14
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    # 14D objective, 0 constraints, X = n-by-14

    tags = {"single_objective", "unconstrained", "continuous", "14D", "extra_imports"}

    def __init__(self):
        super().__init__(
            dim=14,
            num_objectives=1,
            num_constraints=0,
            bounds=[
                (-5, 5),
                (-5, 5),
                (-10, 10),
                (-10, 10),
                (2, 30),
                (0, 2 * np.pi),
                (-5, 5),
                (-5, 5),
                (-10, 10),
                (-10, 10),
                (2, 30),
                (0, 2 * np.pi),
                (-5, 5),
                (-5, 5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        from joblib import Parallel, delayed

        f = PushReward()

        if scaling:
            X = super().scale(X)
        # fx = torch.zeros(X.shape[0],1)

        # Using numpy instead of torch for parallel processing
        fx = np.zeros((X.shape[0], 1))

        # Number of simulations to run in parallel
        n_jobs = 5

        fx[:, 0] = Parallel(n_jobs=n_jobs)(
            delayed(f)(X[i, :].numpy()) for i in range(X.shape[0])
        )

        # for i in range(X.shape[0]):
        #     fx[i,0] = f(X[i,:].numpy())

        # Convert back to torch tensor
        fx = torch.from_numpy(fx)

        return None, fx


##############################
# Helper Functions
##############################
class PushReward:
    def __init__(self):
        # domain of this function
        self.xmin = [
            -5.0,
            -5.0,
            -10.0,
            -10.0,
            2.0,
            0.0,
            -5.0,
            -5.0,
            -10.0,
            -10.0,
            2.0,
            0.0,
            -5.0,
            -5.0,
        ]
        self.xmax = [
            5.0,
            5.0,
            10.0,
            10.0,
            30.0,
            2.0 * np.pi,
            5.0,
            5.0,
            10.0,
            10.0,
            30.0,
            2.0 * np.pi,
            5.0,
            5.0,
        ]

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) + np.linalg.norm(
            np.array(self.gxy2) - np.array(self.sxy2)
        )

    @property
    def dx(self):
        # dimension of the input
        return self._dx

    def __call__(self, argv):
        # returns the reward of pushing two objects with two robots
        rx = float(argv[0])
        ry = float(argv[1])
        xvel = float(argv[2])
        yvel = float(argv[3])
        simu_steps = int(float(argv[4]) * 10)
        init_angle = float(argv[5])
        rx2 = float(argv[6])
        ry2 = float(argv[7])
        xvel2 = float(argv[8])
        yvel2 = float(argv[9])
        simu_steps2 = int(float(argv[10]) * 10)
        init_angle2 = float(argv[11])
        rtor = float(argv[12])
        rtor2 = float(argv[13])

        initial_dist = self.f_max

        world = b2WorldInterface(True)
        _, _, ofriction, odensity, _, hand_shape, hand_size = (
            "circle",
            1,
            0.01,
            0.05,
            0.01,
            "rectangle",
            (1, 0.3),
        )

        base = make_base(500, 500, world)
        body = create_body(
            base, world, "rectangle", (0.5, 0.5), ofriction, odensity, self.sxy
        )
        body2 = create_body(base, world, "circle", 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(
            world, (rx2, ry2), base, init_angle2, hand_shape, hand_size
        )
        (ret1, ret2) = run_simulation(
            world,
            body,
            body2,
            robot,
            robot2,
            xvel,
            yvel,
            xvel2,
            yvel2,
            rtor,
            rtor2,
            simu_steps,
            simu_steps2,
        )

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        return initial_dist - ret1 - ret2


class guiWorld:
    def __init__(self, fps):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1000, 1000
        self.TARGET_FPS = fps
        self.PPM = 10.0  # pixels per meter
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32
        )
        pygame.display.set_caption("push simulator")
        self.clock = pygame.time.Clock()
        self.screen_origin = b2Vec2(
            self.SCREEN_WIDTH / (2 * self.PPM), self.SCREEN_HEIGHT / (self.PPM * 2)
        )
        self.colors = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (163, 209, 224, 255),
        }

    def draw(self, bodies, bg_color=(64, 64, 64, 0)):
        def my_draw_polygon(polygon, body, fixture):
            vertices = [
                (self.screen_origin + body.transform * v) * self.PPM
                for v in polygon.vertices
            ]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            color = self.colors[body.type]
            if body.userData == "obs":
                color = (123, 128, 120, 0)
            if body.userData == "hand":
                color = (174, 136, 218, 0)

            pygame.draw.polygon(self.screen, color, vertices)

        def my_draw_circle(circle, body, fixture):
            position = (self.screen_origin + body.transform * circle.pos) * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            color = self.colors[body.type]
            if body.userData == "hand":
                color = (174, 136, 218, 0)
            pygame.draw.circle(
                self.screen,
                color,
                [int(x) for x in position],
                int(circle.radius * self.PPM),
            )

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle
        # draw the world
        self.screen.fill(bg_color)
        self.clock.tick(self.TARGET_FPS)
        for body in bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        pygame.display.flip()


# this is the interface to pybox2d
class b2WorldInterface:
    def __init__(self, do_gui=True):
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.do_gui = do_gui
        self.TARGET_FPS = 500
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.VEL_ITERS, self.POS_ITERS = 10, 10
        self.bodies = []

        if do_gui:
            self.gui_world = guiWorld(self.TARGET_FPS)
            # raw_input()
        else:
            self.gui_world = None

    def initialize_gui(self):
        if self.gui_world is None:
            self.gui_world = guiWorld(self.TARGET_FPS)
        self.do_gui = True

    def stop_gui(self):
        self.do_gui = False

    def add_bodies(self, new_bodies):
        """add a single b2Body or list of b2Bodies to the world"""
        if isinstance(new_bodies, list):
            self.bodies += new_bodies
        else:
            self.bodies.append(new_bodies)

    def step(self, show_display=True, idx=0):
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
        if show_display and self.do_gui:
            self.gui_world.draw(self.bodies)


class end_effector:
    def __init__(
        self,
        b2world_interface,
        init_pos,
        base,
        init_angle,
        hand_shape="rectangle",
        hand_size=(0.3, 1),
    ):
        world = b2world_interface.world
        self.hand = world.CreateDynamicBody(position=init_pos, angle=init_angle)
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        # forceunit for circle and rect
        if hand_shape == "rectangle":
            rshape = b2PolygonShape(box=hand_size)
            self.forceunit = 30.0
        elif hand_shape == "circle":
            rshape = b2CircleShape(radius=hand_size)
            self.forceunit = 100.0
        elif hand_shape == "polygon":
            rshape = b2PolygonShape(vertices=hand_size)
        else:
            raise Exception("%s is not a correct shape" % hand_shape)

        self.hand.CreateFixture(shape=rshape, density=0.1, friction=0.1)
        self.hand.userData = "hand"

        _ = world.CreateFrictionJoint(
            bodyA=base,
            bodyB=self.hand,
            maxForce=2,
            maxTorque=2,
        )
        b2world_interface.add_bodies(self.hand)

    def set_pos(self, pos, angle):
        self.hand.position = pos
        self.hand.angle = angle

    def apply_wrench(self, rlvel=(0, 0), ravel=0):
        avel = self.hand.angularVelocity
        delta_avel = ravel - avel
        torque = self.hand.mass * delta_avel * 30.0
        self.hand.ApplyTorque(torque, wake=True)

        lvel = self.hand.linearVelocity
        delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
        force = self.hand.mass * delta_lvel * self.forceunit
        self.hand.ApplyForce(force, self.hand.position, wake=True)

    def get_state(self, verbose=False):
        state = (
            list(self.hand.position)
            + [self.hand.angle]
            + list(self.hand.linearVelocity)
            + [self.hand.angularVelocity]
        )
        if verbose:
            print_state = ["%.3f" % x for x in state]
            print
            "position, velocity: (%s), (%s) " % (
                (", ").join(print_state[:3]),
                (", ").join(print_state[3:]),
            )

        return state


def create_body(
    base, b2world_interface, body_shape, body_size, body_friction, body_density, obj_loc
):
    world = b2world_interface.world

    link = world.CreateDynamicBody(position=obj_loc)
    if body_shape == "rectangle":
        linkshape = b2PolygonShape(box=body_size)
    elif body_shape == "circle":
        linkshape = b2CircleShape(radius=body_size)
    elif body_shape == "polygon":
        linkshape = b2PolygonShape(vertices=body_size)
    else:
        raise Exception("%s is not a correct shape" % body_shape)

    link.CreateFixture(
        shape=linkshape,
        density=body_density,
        friction=body_friction,
    )
    _ = world.CreateFrictionJoint(
        bodyA=base,
        bodyB=link,
        maxForce=5,
        maxTorque=2,
    )

    b2world_interface.add_bodies([link])
    return link


def make_base(table_width, table_length, b2world_interface):
    world = b2world_interface.world
    base = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2PolygonShape(box=(table_length, table_width)),
    )

    b2world_interface.add_bodies([base])
    return base


def run_simulation(
    world,
    body,
    body2,
    robot,
    robot2,
    xvel,
    yvel,
    xvel2,
    yvel2,
    rtor,
    rtor2,
    simulation_steps,
    simulation_steps2,
):
    # simulating push with fixed direction pointing from robot location to body location
    desired_vel = np.array([xvel, yvel])
    rvel = b2Vec2(
        desired_vel[0] + np.random.normal(0, 0.01),
        desired_vel[1] + np.random.normal(0, 0.01),
    )

    desired_vel2 = np.array([xvel2, yvel2])
    rvel2 = b2Vec2(
        desired_vel2[0] + np.random.normal(0, 0.01),
        desired_vel2[1] + np.random.normal(0, 0.01),
    )

    tmax = np.max([simulation_steps, simulation_steps2])
    for t in range(tmax + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel, rtor)
        if t < simulation_steps2:
            robot2.apply_wrench(rvel2, rtor2)
        world.step()

    return (list(body.position), list(body2.position))
