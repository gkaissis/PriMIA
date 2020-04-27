import numpy as np


"""
Available schedule plans:
log_linear : Linear interpolation with log learning rate scale
log_cosine : Cosine interpolation with log learning rate scale
"""
class LearningRateScheduler():
    def __init__(self, total_epochs, log_start_lr, log_end_lr, schedule_plan='log_linear'):
        self.total_epochs = total_epochs
        if schedule_plan == 'log_linear':
            self.calc_lr = lambda epoch: np.power(10, ((log_end_lr-log_start_lr)/total_epochs)*epoch + log_start_lr)
        elif schedule_plan == 'log_cosine':
            self.calc_lr = lambda epoch: np.power(10, (np.cos(np.pi*(epoch/total_epochs))/2.+.5)*abs(log_start_lr-log_end_lr) + log_end_lr)
        else:
            raise NotImplementedError('Requested learning rate schedule {} not implemented'.format(schedule_plan))
            
            
    def get_lr(self, epoch):
        if (type(epoch) is int and epoch > self.total_epochs) or (type(epoch) is np.ndarray and np.max(epoch) > self.total_epochs):
            raise AssertionError('Requested epoch out of precalculated schedule')
        return self.calc_lr(epoch)
    
    def adjust_learning_rate(self, optimizer, epoch):
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr