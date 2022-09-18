import haiku as hk

from idiv.models.utils import ConvBlock, residual, MHAttention, prenorm, UpSample, DownSample


class UNet64(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x, is_training):

        inititializer = hk.initializers.VarianceScaling(2.)

        x = hk.Conv2D(self.dim, 3, w_init=inititializer)(x) # 64, 64, 64
        x1 = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x1, is_training)
        x = DownSample(self.dim)(x, is_training) # 32, 32, 64
        x2 = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x2, is_training)
        x = DownSample(self.dim*2)(x, is_training) # 16, 16, 128
        x = residual(ConvBlock())(x, is_training)
        x3 = prenorm(x, residual(MHAttention()))
        x = residual(ConvBlock())(x3, is_training)
        x = DownSample(self.dim*4)(x, is_training) # 8, 8, 256
        x = residual(ConvBlock())(x, is_training)
        x4 = prenorm(x, residual(MHAttention(heads=8)))
        x = residual(ConvBlock())(x4, is_training)
        x = DownSample(self.dim*8)(x, is_training) # 4, 4, 512
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*4)(x, is_training) # 8, 8, 256
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x4, residual(MHAttention(heads=8)))
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*2)(x, is_training) # 16, 16, 128
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x3, residual(MHAttention()))
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim)(x, is_training) # 32, 32, 64
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x2, residual(ConvBlock()), is_training)
        x = UpSample(self.dim)(x, is_training) # 64, 64, 64
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x1, residual(ConvBlock()), is_training)
        
        x = hk.Conv2D(3, 3)(x)

        return x