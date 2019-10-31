from torch import nn

def save_model(network, path):
    torch.save(network.state_dict(),path)
def load_model(network, path, device='cpu', strict=True):
    network.load_state_dict(torch.load(path,map_location=torch.device(device)),strict=strict)
    return network

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MCNET(torch.nn.Module):
    """

    MCNET based on paper from [Zbontar & LeCun, 2015]. This network takes as input two patches of size 11x11 and output the
    likelihood of the two patches being a match.

    Args:
    -   ws: window size (or blocking size) of the input patch
    -   batch_size: batch size
    Returns:
    -   matching cost between the 2 patches, we use 0 for positive match (represent 0 cost to match) and 1 for negative match

    """

    def __init__(self, ws = 11, batch_size=512, load_path = None, strict=True):
        super(MCNET, self).__init__()

        num_feature_map = 112
        kernel_size = 3
        num_hidden_unit = 384
        self.batch_size = batch_size
        self.ws = ws
        self.strict = strict

        self.net = nn.Sequential(
            ############################################################################
            # Student code begin
            ############################################################################
            
            # num_conv_layers: 5
            nn.Conv2d(1, num_feature_map, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            nn.ReLU(),
            nn.Conv2d(num_feature_map, num_feature_map, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            nn.ReLU(),
            nn.Conv2d(num_feature_map, num_feature_map, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            nn.ReLU(),
            nn.Conv2d(num_feature_map, num_feature_map, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            nn.ReLU(),
            nn.Conv2d(num_feature_map, num_feature_map, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            nn.ReLU(),
            
            # concatenate
            Reshape((batch_size, ws**2 * num_feature_map * 2)),
            
            # num_fc_layers: 3
            nn.Linear(ws**2 * num_feature_map * 2,num_hidden_unit),
            nn.ReLU(),
            nn.Linear(num_hidden_unit,num_hidden_unit),
            nn.ReLU(),
            nn.Linear(num_hidden_unit,1),
            nn.Sigmoid()
            
            ############################################################################
            # Student code end
            ############################################################################

        ).to(device)

        self.criterion = nn.BCELoss().to(device)
        if load_path is not None:
          self.net = load_model(self.net,load_path, strict=strict)

    def forward(self, x):

        return self.net(x)



class ExtendedNet(torch.nn.Module):
    """
    For adding layers to a previously defined network.
    
    Args:
    -   orig_model: Model to add layers to
    -   ws: window size (or blocking size) of the input patch
    -   batch_size: batch size 
    -   new_layer_size: number of nodes in fully connected layer or number of channels (filters) in a Conv layer
    -   load_path: file for pretrained model weights
    -   strict: whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict()
    Returns:
    -   matching cost between the 2 patches, we use 0 for positive match (represent 0 cost to match) and 1 for negative match

    """

    def __init__(self, orig_model, ws = 11, batch_size=512, new_layer_size = 384, load_path=None, strict=True):
        super(ExtendedNet, self).__init__()
        
        self.orig_model=orig_model
        self.batch_size = batch_size
        self.ws = ws
        self.strict = strict

        self.net = nn.Sequential(
            ############################################################################
            # Student code begin
            ############################################################################
            
            ############################################################################
            # Student code end
            ############################################################################
        ).to(device)
        
        self.criterion = nn.BCELoss().to(device)
        
        if load_path is not None:
            self.net = load_model(self.net,load_path, strict=strict)     


    def forward(self, x):
        
        return self.net(x)





# def gen_patch(image, x, y, ws = 11):
#     """
#     function to return a patch of size ws at a specific location of the image.
#     x, y in this case is a top left corner of the patch, for example if x,y is (0,0)
#     you should return a patch over (0,0) and (ws,ws)
    
#     For corner case, you can pad the output with zeros such that we always have 
#     (channel, ws, ws) dimension output
    
#     Args:
#     -   image: image of type Tensor with dimension (channel, width, height)
#     -   x: x location in the image 
#     -   y: y location in the image
#     -   ws: window size or block size of the patch we want
#     Returns:
#     -   patch: a patch of size (channel, ws, ws) of type Tensor

#     """
#     ############################################################################
#     # Student code begin
#     ############################################################################
    
#     x = int(x)
#     y = int(y)
    
#     patch = image[:,x:x+ws,y:y+ws]
    
#     if patch.shape[1] != ws or patch.shape[2] != ws:
#       padded_patch = torch.zeros(image.shape[0],ws,ws)
#       padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
#       patch = padded_patch
    
#     return patch
  
#     ############################################################################
#     # Student code end
#     ############################################################################