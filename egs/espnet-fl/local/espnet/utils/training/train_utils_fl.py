import chainer
import logging


def check_early_stop(trainer, epochs):
    """Checks if the training was stopped by an early stopping trigger and warns the user if it's the case

    :param trainer: The trainer used for training
    :param epochs: The maximum number of epochs
    """
    end_epoch = trainer.updater.get_iterator('main').epoch
    if end_epoch < (epochs - 1):
        logging.warning("Hit early stop at epoch " + str(
            end_epoch) + "\nYou can change the patience or set it to 0 to run all epochs")


def set_early_stop(trainer, args, is_lm=False):
    """Sets the early stop trigger given the program arguments

    :param trainer: The trainer used for training
    :param args: The program arguments
    :param is_lm: If the trainer is for a LM (epoch instead of epochs)
    """
    patience = args.patience
    criterion = args.early_stop_criterion
    epochs = args.epoch if is_lm else args.epochs
    mode = 'max' if 'acc' in criterion else 'min'
    if patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=criterion,
                                                                              mode=mode,
                                                                              patients=patience,
                                                                              max_trigger=(epochs, 'epoch'))


def set_iter_stop(trainer,args):
    """ Sets early stop trigger for given number of iterations

    :param trainer: The trainer used for training
    :param args: The program arguments
    """
    #print(trainer.updater.iteration+args.iterval)

    trainer.stop_trigger = chainer.training.triggers.IntervalTrigger(trainer.updater.iteration+args.iterval, 'iteration')
    #print("SET ITERSTOP")


def set_epoch_stop(trainer,args):
    """ Sets early stop trigger for given number of iterations

    :param trainer: The trainer used for training
    :param args: The program arguments
    """
    trainer.stop_trigger = chainer.training.triggers.IntervalTrigger(args.epochs+1, 'epoch')
    #print("SET EPOCHSTOP", trainer.stop_trigger)

def set_next_epoch_stop(trainer,args):
    """ Sets early stop trigger for given number of iterations

    :param trainer: The trainer used for training
    :param args: The program arguments
    """
    trainer.stop_trigger = chainer.training.triggers.IntervalTrigger(trainer.updater.epoch+1, 'epoch')
    #print(trainer.updater.epoch+1)
    #print("SET EPOCHSTOP", trainer.stop_trigger)

