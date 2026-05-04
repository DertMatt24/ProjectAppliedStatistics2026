from loader.eeg_recording import EEGLoader

if __name__ == '__main__':

    nights = []
    for i in range(1, 40):
        for j in range(1, 2+1):
            nights.append((i, j))
    for p_data in nights:
        print(p_data)
        print(EEGLoader.load('C:/Users/picul/Videos/Applied/Dataset_Full/polysomnographics/', *p_data).data.shape)
