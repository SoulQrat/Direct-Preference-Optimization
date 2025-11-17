import torch.nn.functional as F

def log_prob(logits, mask, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (log_probs * mask).sum(dim=-1)

def DPOLoss(out_model_w, out_model_l, out_ref_model_w, out_ref_model_l, mask, labels_chosen, labels_rejected, beta):

    mask = mask.to('cuda:0')
    labels_chosen = labels_chosen.to('cuda:0')
    labels_rejected = labels_rejected.to('cuda:0')

    prob_model_w = log_prob(out_model_w.logits.to('cuda:0'), mask, labels_chosen)
    prob_model_l = log_prob(out_model_l.logits.to('cuda:0'), mask, labels_rejected)
    prob_rew_model_w = log_prob(out_ref_model_w.logits.to('cuda:0'), mask, labels_chosen)
    prob_rew_model_l = log_prob(out_ref_model_l.logits.to('cuda:0'), mask, labels_rejected)

    return -F.logsigmoid(beta * ((prob_model_w - prob_rew_model_w) - (prob_model_l - prob_rew_model_l))).mean(dim=-1)