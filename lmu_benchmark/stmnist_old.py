
def load_digit_old(digit=8,n=0): # Don't need this function anymore
    # Change this to your dataset path
    dataset_path = 'C:/Users/karth/Desktop/Telluride/ST-MNIST/'
    path = dataset_path+'data_submission/'+str(digit)+'/'+str(digit)+'_ch0_'+str(n)+'_spiketrain.mat'
    mat = scipy.io.loadmat(path)
    mdata = mat['spiketrain']
    mdata = np.array(mdata)
    #print(mdata.shape[0])
    spikes=mdata[0:100,:]
    times=mdata[100,:]
    print(spikes.shape)
    print(times.shape)
    spikes_plus=np.zeros(spikes.shape[1])
    spikes_minus=np.zeros(spikes.shape[1])
    for i in range(spikes.shape[1]):
        for j in range(spikes.shape[0]):
            if spikes[j,i]==1:
                spikes_plus[i]=j
            if spikes[j,i]==-1:
                spikes_minus[i]=j
    return times,spikes_plus,spikes_minus

def visualize_digit(times,spikes_plus,spikes_minus):
    yp1 = (spikes_plus)%10
    yp10 = (spikes_plus-yp1)/10
    plt.plot(yp10,10-yp1,'ro')
    plt.show()

def visualize_spiketrain(times,spikes_plus,spikes_minus):
    plt.plot(times,spikes_plus,'ro')
    plt.plot(times,spikes_minus,'bo')
    plt.grid()
    plt.show()

def convert_stmnist():
    dataset_path = 'C:/Users/karth/Desktop/Telluride/ST-MNIST/'
    for digit in range(10):
        for n in range(690):
            spikedata=[digit]
            path = dataset_path+'data_submission/'+str(digit)+'/'+str(digit)+'_ch0_'+str(n)+'_spiketrain.mat'
            mat = scipy.io.loadmat(path)
            mdata = mat['spiketrain']
            mdata = np.array(mdata)
            spikes=mdata[0:101,:]
            times=mdata[100,:]
            print(spikes.shape[1])
            print(spikes.shape[0])
            for i in range(min(12000,spikes.shape[1])):
                for j in range(spikes.shape[0]):
                    if spikes[j,i]==1:
                        #spikedata[n,i,digit]=j
                        spikedata.append(j)
                    if spikes[j,i]==-1:
                        #spikedata[n,i,digit]=-j
                        spikedata.append(-j)
            with open('spikedata.csv','a') as f:
                    writer_object = writer(f)
                    writer_object.writerow(spikedata)
                    f.close()
    print("done")
