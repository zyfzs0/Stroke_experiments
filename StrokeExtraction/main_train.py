from train_SDNet import TrainSDNet
from train_SegNet import TrainSegNet
from train_ExtractNet import TrainExtractNet

class MainTrain():
    def __init__(self, dataset='RHSEDB'):
        '''
        Select the dataset to train from ['CCSEDB', 'RHSEDB']
        '''
        self.dataset = dataset
        self.train_sdnet = TrainSDNet(save_path='out0/7_SDNet_updown_', dataset=dataset)
        self.train_segnet = TrainSegNet(save_path='out0/7_SegNet_updown_'+dataset)
        self.train_extractnet = TrainExtractNet(save_path='out0/7_ExtractNet_updown_'+dataset, segNet_save_path='out0/7_SegNet_updown_'+dataset)

    def train(self):
        # train SDNet
        self.train_sdnet.train_model(epochs=20, init_learning_rate=0.0001, batch_size=16)
        print('SDNet training has been completed')
        # get prior information and other data for SegNet and ExtractNet
        self.train_sdnet.calculate_prior_information_and_qualitative(save_path='dataset_forSegNet_ExtractNet7_updown_'+self.dataset)
        print('calculating prior information has been completed')
        # train SegNet
        self.train_segnet.train_model(epochs=10, init_learning_rate=0.0001, batch_size=16, dataset_path='dataset_forSegNet_ExtractNet7_updown_'+self.dataset)
        print('SegNet training has been completed')
        #train ExtractNet
        self.train_extractnet.train_model(epochs=20, init_learning_rate=0.0001, batch_size=16, dataset='dataset_forSegNet_ExtractNet7_updown_'+self.dataset)
        print('ExtractNet training has been completed')


if __name__ == '__main__':
    train_ = MainTrain(dataset='self')
    train_.train()
