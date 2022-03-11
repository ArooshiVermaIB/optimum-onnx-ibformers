from optuna import Study
from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl

from ibformers.trainer.arguments import EnhancedTrainingArguments


class IbCallbackHandler(CallbackHandler):
    def on_hyperparam_search_start(self, args: EnhancedTrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_hyperparam_search_start", args, state, control)

    def on_hyperparam_search_end(
        self, args: EnhancedTrainingArguments, state: TrainerState, control: TrainerControl, study: Study
    ):
        return self.call_event("on_hyperparam_search_end", args, state, control, study=study)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            # we ignore callbacks without the event handled - this allows us to implement custom callback events
            if not hasattr(callback, event):
                continue
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control

    @classmethod
    def from_callback_handler(cls, callback_handler: CallbackHandler):
        return cls(
            callbacks=callback_handler.callbacks,
            model=callback_handler.model,
            tokenizer=callback_handler.tokenizer,
            optimizer=callback_handler.optimizer,
            lr_scheduler=callback_handler.lr_scheduler,
        )
