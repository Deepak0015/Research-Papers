import torch




def cal_loss_batch(input_batch , target_batch , model:torch.nn.Module , device:torch.device):
    input_batch , target_batch = input_batch.to(device) , target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(   logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader , model , device , num_batches = None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches  = min(num_batches , len(data_loader))
    for i , (inputs , target) in enumerate(data_loader):
        if i < num_batches:
            loss  =  cal_loss_batch(inputs , target , model , device)

            total_loss +=loss.item()

        else:
            break

        return total_loss  / num_batches
    
