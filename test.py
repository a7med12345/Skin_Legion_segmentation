import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.measure import compare_psnr
import pybm3d
from util import util
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt



if __name__ == '__main__':


    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test
    psnr=[]

    for i, data in enumerate(dataset):
        #if i >= opt.num_test:
            #break
        model.set_input(data)
        model.test()

        #print("a")

        #loss_psnr = compare_psnr(model.real_B.cpu().numpy()[0].transpose(2,1,0),model.real_A.cpu().numpy()[0].transpose(2,1,0))
        #loss_psnr_filter = compare_psnr(model.real_T.cpu().numpy()[0].transpose(2,1,0),model.real_C.cpu().numpy()[0].transpose(2,1,0))
        #loss_psnr_before = compare_psnr(model.real_T.cpu().numpy()[0].transpose(2,1,0),model.real_A.cpu().numpy()[0].transpose(2,1,0))

        #print(model.real_B.cpu().numpy()[0].transpose(2,1,0).shape)
        #out = model.real_A1.cpu().numpy()[0].transpose(2,1,0)
        #print(estimate_sigma(util.tensor2im(model.real_A1),multichannel=True)[0])
        #bm3d_out = pybm3d.bm3d.bm3d(util.tensor2im(model.real_A1),estimate_sigma(util.tensor2im(model.real_A1),multichannel=True)[0])
        #plt.imshow(bm3d_out)
        #plt.show()
        #loss_psnr_bm3d = compare_psnr(util.tensor2im(model.real_B), bm3d_out)
        #loss_psnr_bm3d=0
        #psnr.append([loss_psnr_n, loss_psnr_c, loss_psnr_before])
        #psnr.append([loss_psnr_n,loss_psnr_before])
        #print("denoised ",loss_psnr_n)
        #print("bilateral filter ",loss_psnr_filter)
        #print("before ",loss_psnr_before)
        #print(loss_psnr)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        #if i%5==0:
            #print('processing (%04d)-th image... %s' % (i, img_path))
            #print("n2n",psnr[i][0])
            #print("n2c",psnr[i][1])
            #print("before", psnr[i][2])
            #print("bm3d", psnr[i][3])
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    #with open('/home/ahmed/Pictures/n2n_epoch_'+opt.epoch+'.txt', 'w') as f:
        #for item in psnr:
            #f.write("%s\n" % item[0])

    #with open('/home/ahmed/Pictures/n2c_epoch_'+opt.epoch+'.txt', 'w') as f:
        #for item in psnr:
            #f.write("%s\n" % item[1])

    #print(sum(psnr[0])/len(psnr[0]))
    #print(sum(psnr[1])/len