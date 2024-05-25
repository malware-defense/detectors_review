import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attack_label(model, data_loader, adv_init_mag, adv_lr, perturbation_steps, detect_type):
    model.eval()
    pbar = tqdm(data_loader)
    perturbation_step_list = list()
    label_not_flip_rate = list()
    criteria_all = None

    for step in range(perturbation_steps):
        perturbation_step_list.append(step)
        init_y_pred_sum, attack_after_y_pred_sum, criteria_sum = None, None, None
        for model_inputs, y_label in pbar:
            model.zero_grad()

            # init pred
            init_model_outputs = model(model_inputs)
            init_y_pred = init_model_outputs.max(1)[1]

            # pred after perturbation
            perturbed_model_inputs = perturb_as_denfense(model, model_inputs, init_y_pred, step, adv_lr, adv_init_mag)
            perturbed_model_outputs = model(perturbed_model_inputs)
            attack_after_y_pred = perturbed_model_outputs.max(1)[1]

            attack_losses = F.cross_entropy(perturbed_model_outputs, init_y_pred, reduction='none')
            if detect_type == 'loss':
                criteria = attack_losses
            elif detect_type == 'logit':
                criteria = torch.norm(perturbed_model_outputs - init_model_outputs, p=2, dim=1)

            if init_y_pred_sum is None:
                init_y_pred_sum = init_y_pred
                attack_after_y_pred_sum = attack_after_y_pred
                criteria_sum = criteria
            else:
                init_y_pred_sum = torch.cat((init_y_pred_sum, init_y_pred), dim = 0)
                attack_after_y_pred_sum = torch.cat((attack_after_y_pred_sum, attack_after_y_pred), dim = 0)
                criteria_sum = torch.cat((criteria_sum, criteria), dim = 0)

        not_flip_rate = sum(init_y_pred_sum == attack_after_y_pred_sum) / len(init_y_pred_sum)
        # print(not_flip_rate)
        label_not_flip_rate.append(not_flip_rate)

        if criteria_all is None:
            criteria_all = criteria_sum.unsqueeze(0)
        else:
            criteria_all = torch.cat((criteria_all, criteria_sum.unsqueeze(0)), dim = 0)
    return perturbation_step_list, label_not_flip_rate, criteria_all



def perturb_as_denfense(model, model_inputs, init_pred_labels, step, adv_lr, adv_init_mag):
    if step == 0:
        return model_inputs

    #initialize delta
    delta = torch.zeros_like(model_inputs).uniform_(-1, 1)
    dim = torch.from_numpy(np.array(model_inputs.shape[1]))
    magnitude = adv_init_mag / torch.sqrt(dim)
    delta = (delta * magnitude.view(-1, 1))

    for sub_step in range(step):
        delta.requires_grad_()
        model_outputs = model(model_inputs + delta)
        losses = F.cross_entropy(model_outputs, init_pred_labels)
        loss = torch.mean(losses)
        loss = loss / step
        loss.backward()

        del loss

        delta_grad = delta.grad.clone().detach()
        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1)
        denorm = torch.clamp(denorm, min=1e-8)
        delta = (delta + adv_lr * delta_grad / denorm).detach()

    return model_inputs + delta
