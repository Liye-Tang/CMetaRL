import torch
import onnx
import onnxruntime as ort
import torchvision.transforms as transform
import onnxmltools
import os
import torch.nn as nn
from models.policy import Policy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert():

    # Load trained policy

    load_path = "./logs/logs_MultiGoalEnv-v0/varibad_74__10:01_05:46:27"
    
    iter = 499

    policy = torch.load(os.path.join(load_path, 'models/policy{}.pt'.format(iter)))
    
    
    transformed_policy = TransPolicy()
    
    transformed_policy = clone_policy(transformed_policy, policy)
    
    # transformed_policy.forward = forward
    

    transformed_policy.eval()

    example = torch.rand(1, 66)
    
    print(transformed_policy(example))

    output_onnx_model = './transform_network011000.onnx'

    torch.onnx.export(transformed_policy, example, output_onnx_model, input_names=['input'], output_names=['output'], verbose=True, opset_version=11)

    #

    # # traced_script_module = torch.jit.trace(networks.policy, example)

    # # traced_script_module.save('../transform_pt_network/network_1228.pt')





    # pt_net = onnx.load('../transform_pt_network/network_fhadp_1230.onnx')

    # onnx.checker.check_model(pt_net)



    ort_session = ort.InferenceSession(output_onnx_model)

    example1 = np.zeros((1, 66)).astype(np.float32)

    ego_state = np.array([-0.3, 0, 0, -0.406337, 0.00105598, 0])

    ref_state1 = np.array([0] * 20)
    ref_state2 = np.array([i * 0.05 for i in range(20)])
    ref_stata3 = np.array([0] * 20)

    example1 = np.concatenate((ego_state, ref_state1, ref_state2, ref_stata3)).astype(np.float32)
    example1 = np.expand_dims(example1, 0)

    inputs = {ort_session.get_inputs()[0].name: example1}

    outputs = ort_session.run(None, inputs)

    # outputs = ort_session.run(None, {"input":np.random.randn(1, 66).astype(np.float32)})



    print(outputs)

    action = transformed_policy(torch.tensor(example1))

    print(action)

    # print(pt_net(example))
    

def clone_policy(a, b):
    for k, v in b.__dict__.items():
        setattr(a, k, v)
        
    for k, v in b.__class__.__dict__.items():
        if k.startswith('__') or k == 'forward' or k == 'act':
            continue
        
        setattr(a.__class__, k, getattr(b.__class__, k))
    
    return a


class TransPolicy(nn.Module):
    def __init__(self):
        super(TransPolicy, self).__init__()
        
    def forward(self, input):
            # handle inputs (normalise + embed)
        state = input
        # latent = None
        # belief = None
        # task = None

        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            if self.use_state_encoder:
                state = self.state_encoder(state)
        else:
            state = torch.zeros(0, ).to(device)
        # if self.pass_latent_to_policy:
        #     if self.norm_latent:
        #         latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
        #     if self.use_latent_encoder:
        #         latent = self.latent_encoder(latent)
        # else:
        #     latent = torch.zeros(0, ).to(device)
        # if self.pass_belief_to_policy:
        #     if self.norm_belief:
        #         belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
        #     if self.use_belief_encoder:
        #         belief = self.belief_encoder(belief.float())
        # else:
        #     belief = torch.zeros(0, ).to(device)
        # if self.pass_task_to_policy:
        #     if self.norm_task:
        #         task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
        #     if self.use_task_encoder:
        #         task = self.task_encoder(task.float())
        # else:
        #     task = torch.zeros(0, ).to(device)

        # concatenate inputs
        inputs = state

        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        
        dist = self.dist(hidden_actor)
        
        action = dist.mean
        
        return action


convert()