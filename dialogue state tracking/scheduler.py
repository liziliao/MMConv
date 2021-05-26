from timm.scheduler import StepLRScheduler as StepLRSched


class StepLRScheduler(StepLRSched):
    def __init__(
        self,
        optimizer,
        decay_t,
        decay_rate=1,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
        last_update=-1
    ):
        super().__init__(
            optimizer,
            decay_t,
            decay_rate=decay_rate,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize
        )
        self.last_update = last_update

    def step(self, metric=None):
        self.last_update += 1
        return super().step(self.last_update, metric=metric)

