import paddle
from models.model_stages import CPNet
import paddleseg.transforms as T
from paddleseg.datasets import Cityscapes
from paddle.optimizer.lr import PolynomialDecay
from paddleseg.models.losses import CrossEntropyLoss
from loss.affinityloss import AffinityLoss
from tool.train import train

backbonepath = None
model = CPNet(proir_size=96, am_kernel_size=11, groups=1, prior_channels=256, pretrained=backbonepath)

transform = [
    T.ResizeStepScaling(0.5, 2.0, 0.25),
    T.RandomHorizontalFlip(),
    T.RandomPaddingCrop(crop_size=[768, 768]),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
train_dataset = Cityscapes(
    dataset_root='',
    transforms=transform,
    mode='train'
)

transform_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

val_dataset = Cityscapes(
    dataset_root='',
    transforms=transform_val,
    mode='val'
)

base_lr = 0.01

lr = PolynomialDecay(
    learning_rate=base_lr,
    decay_steps=60000,
    power=0.9,
    verbose=True
)

optimizer = paddle.optimizer.Momentum(lr,
                                      parameters=model.parameters(),
                                      momentum=0.9,
                                      weight_decay=5.0e-4)

losses = {}

losses['types'] = [
    CrossEntropyLoss(),
    CrossEntropyLoss(),
    AffinityLoss()
]
losses['coef'] = [1, 1, 0.4]

if __name__ == '__main__':
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_scales=1,
        aug_eval=True,
        optimizer=optimizer,
        save_dir='output',
        iters=60000,
        batch_size=5,
        save_interval=200,
        resume_model='',
        log_iters=10,
        num_workers=8,
        losses=losses,
        use_vdl=False
    )
