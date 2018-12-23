# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


from utils import get_subwindow_tracking


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)//total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    
def tracker_score(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
    
    return score.max()


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)
    # print(score.shape)
    
    # score_var = (score.permute(1, 2, 3, 0).contiguous().view(2, -1)).data[1, :].cpu().numpy()

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score
    # score_var = score_var * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)
    # score_var = score_var * (1 - p.window_influence) + window * p.window_influence

    
    
    ##################################
    ##################################
    # print(score_var.shape)
    # print(score.shape)
    # print(best_pscore_id)
    arg_pscore = np.argsort(pscore)
    # best_pscore_id = arg_pscore[-2]
    # print(arg_pscore[-1])
    # print(pscore[arg_pscore[-1]])
    # print(delta[:, arg_pscore[-1]])
    
    # topk_anchor = arg_pscore[-2:]
    # topk_pscore = score_var[topk_anchor]
    # print(topk_pscore)
    # topk_delta = delta[:, topk_anchor]
    # delta_average = np.average(topk_delta, axis=1, weights=topk_pscore)
    # print(delta_average)
    '''
    label = np.zeros(score.shape)
    label[best_pscore_id] = 1
    clabel = torch.from_numpy(label).cuda()
    closs = nn.CrossEntropyLoss()(coutput, clabel)
    '''
    

    target = delta[:, best_pscore_id] / scale_z
    # target = delta_average / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net, gtbox):
    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    state['ctr'] = 0

    if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        p.instance_size = 287  # small object big search region
    else:
        p.instance_size = 271

    p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    # plt.imshow(z_crop.numpy()[0])
    # plt.show()

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())
    
    # wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    # hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    '''
    clabel = np.zeros([5, 17, 17]) - 100
    my_dataset = MyDataset(root_dir = "./dataset/OTB")
    pos, neg = my_dataset._get_64_anchors(gtbox)
    for i in range(len(pos)):
        clabel[pos[i, 2], pos[i, 0], pos[i, 1]] = 1
    for i in range(len(neg)):
        clabel[neg[i, 2], neg[i, 0], neg[i, 1]] = 0
    for cl in clabel:
        plt.imshow(cl, cmap="rainbow")
        plt.show()
    clabel = torch.Tensor(clabel).long()
    print(clabel.shape)
    
    '''
    
    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    # x_show = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans, out_mode="image")
    # plt.grid(False)
    # plt.imshow(x_show)
    # plt.show()
    
    label = np.zeros(im.shape)
    x_low, x_high = target_pos[0] - target_sz[0]/2, target_pos[0] + target_sz[0]/2
    y_low, y_high = target_pos[1] - target_sz[1]/2, target_pos[1] + target_sz[1]/2
    x_low, x_high, y_low, y_high = int(x_low), int(x_high), int(y_low), int(y_high)
    label[y_low:y_high,x_low:x_high,:] = 2
    label = (get_subwindow_tracking(label, target_pos, p.instance_size, round(s_x), 0, out_mode="image"))
    label = cv2.split(label)[0]
    if p.instance_size == 271:
        label = cv2.resize(label, (19, 19))
    else:
        label = cv2.resize(label, (21, 21))
    
    # plt.grid(False)
    # plt.imshow(label, cmap="rainbow")
    # plt.show()
    
    label = torch.Tensor([(2-label)]*5 + [label]*5).unsqueeze(0)
    
    net.make_at(x_crop.cuda(), label.cuda())
    
    
    '''
    # label = np.zeros(x_crop.shape[-2:])
    label = np.zeros((im.shape[0]))
    x_low, x_high = label.shape[0]//2 - target_sz[0]//2, label.shape[0]//2 + target_sz[0]//2
    y_low, y_high = label.shape[1]//2 - target_sz[1]//2, label.shape[1]//2 + target_sz[1]//2
    x_low, x_high, y_low, y_high = int(x_low), int(x_high), int(y_low), int(y_high)
    # print(x_low)
    label[y_low:y_high,x_low:x_high,:] = 1
    label = (get_subwindow_tracking(label, target_pos, p.instance_size, round(s_x), avg_chans, out_mode="image"))
    #label = cv2.resize(label, (19, 19))
    plt.imshow(label)
    plt.show()
    #label_neg = 1-label
    label = torch.Tensor([(1-label)/3]*5 + [label]*5).unsqueeze(0)
    print(label.shape)
    '''
    
    
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    
    '''
    x_low, x_high = target_pos[0] - target_sz[0] // 2, target_pos[0] + target_sz[0] // 2
    y_low, y_high = target_pos[1] - target_sz[1] // 2, target_pos[1] + target_sz[1] // 2
    x_low, x_high, y_low, y_high = int(x_low), int(x_high), int(y_low), int(y_high)
    # print(im.shape)
    z_gt = im[y_low:y_high, x_low:x_high]
    # z_gt = cv2.resize(z_gt, (127, 127))
    plt.imshow(z_gt)
    plt.show()
    z_gt = Variable(im_to_torch(z_gt).unsqueeze(0))
    resnet.template(z_gt.cuda())
    '''
    return state
    


def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    ctr = state['ctr']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['ctr'] = ctr+1
    
    if ctr % 50 == 4:
        label = np.zeros(im.shape)
        x_low, x_high = target_pos[0] - target_sz[0]/2, target_pos[0] + target_sz[0]/2
        y_low, y_high = target_pos[1] - target_sz[1]/2, target_pos[1] + target_sz[1]/2
        x_low, x_high, y_low, y_high = int(x_low), int(x_high), int(y_low), int(y_high)
        label[y_low:y_high,x_low:x_high,:] = 2
        label = (get_subwindow_tracking(label, target_pos, p.instance_size, round(s_x), 0, out_mode="image"))
        label = cv2.split(label)[0]
        if p.instance_size == 271:
            label = cv2.resize(label, (19, 19))
        else:
            label = cv2.resize(label, (21, 21))
        
        # plt.grid(False)
        # plt.imshow(label, cmap="rainbow")
        # plt.show()
        
        label = torch.Tensor([(2-label)]*5 + [label]*5).unsqueeze(0)
        
        net.make_at_small(x_crop.cuda(), label.cuda())
    '''
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z)) - 20
    step = 5
    fine_tune_size = 200
    x_crop = Variable(get_subwindow_tracking(im, target_pos, fine_tune_size, s_z, avg_chans).unsqueeze(0))
    x_show = x_crop.permute(0,2,3,1).numpy()[0]
    # print(x_show.shape)
    plt.imshow(x_show)
    plt.show()
    '''
    '''
    x_low, x_high = target_pos[0] - target_sz[0] // 2, target_pos[0] + target_sz[0] // 2
    y_low, y_high = target_pos[1] - target_sz[1] // 2, target_pos[1] + target_sz[1] // 2
    x_low, x_high, y_low, y_high = int(x_low), int(x_high), int(y_low), int(y_high)
    for i in range(10, 20):
        x_part = im[y_low-i:y_high+i, x_low-i:x_high+i]
        plt.imshow(x_part)
        plt.show()
        
        x_part = Variable(im_to_torch(x_part).unsqueeze(0))
        score = resnet.forward(x_part.cuda())
        print(score, score.cpu().numpy().max())
    '''
    return state
