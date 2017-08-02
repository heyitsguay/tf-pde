import ctypes
import noise
import numpy as np
import pyglet
import tensorflow as tf


# noinspection PyPep8Naming
def main():
    height = 768
    width = 1366

    screen_height = 1080
    screen_width = 1920

    window_x0 = screen_width / 2 - width / 2
    window_y0 = screen_height / 2 - height / 2

    window_style = pyglet.window.Window.WINDOW_STYLE_BORDERLESS

    formats = 'G'

    bound = 0.1
    bound_scale = 255 / (2. * bound)

    # Keep global variables that need to be modified in a dict to get around
    # some Python 2 limitations
    var = {
        'display_idx': 0
    }

    def to_c(arr):
        # arr = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)
        # noinspection PyUnresolvedReferences
        ab = (bound_scale * (arr + bound)).astype('uint8')
        return ab.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def update(dt):
        image_data.blit(0, 0)

    pyglet.clock.schedule(update)

    device = '/gpu:1'
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=graph,
                      config=config)

    def make_kernel(arr):
        arr = np.asarray(arr)
        arr = arr.reshape(list(arr.shape) + [1, 1])
        return tf.constant(arr, dtype=1)

    def conv(x, k):
        x = tf.expand_dims(tf.expand_dims(x, 0), -1)
        y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
        return y[0, :, :, 0]

    def laplace(x):
        # noinspection PyTypeChecker
        laplace_k = make_kernel([[0.5, 1., 0.5],
                                 [1., -6., 1.],
                                 [0.5, 1., 0.5]])
        return conv(x, laplace_k)

    def perlin_array(h, w, base):
        def perlin(m, n):
            return noise.pnoise2(m / float(0.01*h),
                                 n / float(0.01*w),
                                 octaves=1,
                                 repeatx=width,
                                 repeaty=height,
                                 base=base)

        arr = np.zeros((h, w))

        for y in range(100):
            for x in range(100):
                arr[h//2 + y, w//2 + x] = max(-bound, min(bound, (perlin(y,
                                                                         x))))

        return arr

    u_init = np.zeros([height, width, 3], dtype=np.float32)
    u_init[:, :, 0] = perlin_array(height, width, 0)
    # u_init[:, :, 1] = perlin_array(height, width, 17)

    ut_init = np.zeros([height, width, 3], dtype=np.float32)

    image_data = pyglet.image.ImageData(width,
                                        height,
                                        formats,
                                        to_c(u_init[:, :, 0]))

    with sess.as_default(), graph.as_default(), graph.device(device):
        eps = tf.placeholder_with_default(0.01, shape=[])
        diffuse = tf.placeholder_with_default([0.01, 0.01, 0.01], shape=[3])
        damp = tf.placeholder_with_default([0., 0., 0.], shape=[3])
        react = tf.placeholder_with_default([0.1, 1., 0.], shape=[3])
        tau = tf.placeholder_with_default(0.1, shape=[])
        sigma = tf.placeholder_with_default(0.5, shape=[])
        ell = tf.placeholder_with_default(1., shape=[])
        kappa = tf.placeholder_with_default(1., shape=[])
        U = tf.Variable(u_init)
        Ut = tf.Variable(ut_init)

        ''' PDE computation goes here '''

        U_ = tf.maximum(-bound, tf.minimum(bound, U + eps * Ut))

        U1_ = U[:, :, 0]
        U2_ = U[:, :, 1]
        U3_ = U[:, :, 2]

        Ut1_ = Ut[:, :, 0]
        Ut2_ = Ut[:, :, 1]
        Ut3_ = Ut[:, :, 2]

        # noinspection PyTypeChecker
        # R1_ = ell * U1_ - tf.pow(U1_, 3) - kappa
        R1_ = -tf.multiply(tf.multiply(U1_, (U1_ + ell)), (U1_ - 1))

        Ut_ = tf.stack([

            diffuse[0] * laplace(U1_) +
            react[0] * R1_ - sigma * U2_,

            tau * (diffuse[1] * laplace(U2_) + U1_ - U2_),

            0 * U3_
        ],
            axis=2)

        step = tf.group(
            U.assign(U_),
            Ut.assign(Ut_))

    def thread_fn(dt):
        with graph.as_default(), graph.device(device):
            sess.run(step, feed_dict={eps: 0.0167,
                                      diffuse: [0.00015, 0.0001, 0.1],
                                      damp: [0., 0., 0.],
                                      react: [1., 1., 0.],
                                      tau: 1.,
                                      sigma: 1.,
                                      ell: -2.,
                                      kappa: -0.05
                                      })
            u_array = U.eval(session=sess)
            image_data.set_data(formats,
                                len(formats) * width,
                                to_c(u_array[:, :, var['display_idx']]))

    pyglet.clock.schedule_interval(thread_fn, 1. / 80.)

    with graph.as_default(), graph.device(device):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

    window = pyglet.window.Window(width,
                                  height,
                                  fullscreen=False,
                                  style=window_style,
                                  vsync=False)
    window.set_location(1920 + window_x0, window_y0)
    fps_display = pyglet.clock.ClockDisplay()

    @window.event
    def on_draw():
        window.clear()
        image_data.blit(0, 0)
        fps_display.draw()

    @window.event
    def on_show():
        pass

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            var['display_idx'] = 1 - var['display_idx']

    pyglet.app.run()


if __name__ == '__main__':
    main()
