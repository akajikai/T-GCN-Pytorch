import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class TGCN(nn.Module):


    def __init__(self, num_nodes, num_features, gru_units, seq_len,
                 pre_len):

        super(TGCN, self).__init__()

        self._nodes = num_nodes  # 图的节点数
        self._units = gru_units  # gru个数
        self._features = num_features
        self._len = pre_len
        self.grucell1 = nn.GRUCell(input_size=num_features,
                                   hidden_size=gru_units, bias=True)
        self.linear = nn.Linear(in_features=gru_units,
                                out_features=pre_len)


    def forward(self, A_hat, X, state):
        """
        :param X: Input data of shape (seq_len, batch_size, num_nodes).
        :param state: state of shape (batch_size, num_nodes, gru_units).
        :param A_hat: Normalized adjacency matrix(num_nodes, num_nodes).
        """

        ##hx = torch.zeros(X.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        for i in range(X.size(0)):
            Ax = torch.einsum("ij,jk->ki", [A_hat, X[i].permute(1, 0)])
            state = torch.einsum("ij,jkl->kil", [A_hat, state.permute(1, 0, 2)])
            Ax = Ax.reshape(-1, self._features)

            state = state.reshape(-1, self._units)
            state = self.grucell1(Ax, state)
            state = state.reshape(-1, self._nodes, self._units)

        ##state: state of shape (batch_size, num_nodes, gru_units).

        output = state.reshape(-1, self._units)
        ## output:(batch * num_nodes,gru_units)

        output = self.linear(output)

        ## output:(batch * num_nodes,pre_len)
        output = output.reshape(-1, self._nodes, self._len)
        ## output:(batch,num_nodes,pre_len)
        output = output.permute(0,2,1)

        return output


class GCNBlock(nn.Module):

    def __init__(self, num_nodes, gru_units,output_size):

        super(GCNBlock, self).__init__()
        self._nodes = num_nodes
        self._outputsize = output_size
        self.linear = nn.Linear(in_features=gru_units+1,
                                out_features=output_size,bias=True)



    def forward(self,A_hat, X, state):
        """
        :param X: Input data of shape (batch_size, num_nodes).
        :param state: state of shape (batch_size, num_nodes, gru_units).
        :param A_hat: Normalized adjacency matrix(num_nodes, num_nodes).
        """
        X = X.unsqueeze(dim=2)
        ## inputs:(batch,num_nodes,1)

        x_s = torch.cat((X, state), axis=2)
        ## x_s:(batch,num_nodes,gru_units+1)

        input_size = x_s.shape[2]
        # input_size == gru_units+1

        x0 = x_s.permute(1, 2, 0)
        ## x0:(num_nodes,input_size,-1)
        ## x0:(num_nodes,gru_units+1,batch)

        x0 = x0.reshape(self._nodes, -1)
        ## x0:(num_nodes,input_size*batch)
        ## x0:(num_nodes,(gru_units+1)*batch)

        x1 = torch.matmul(A_hat, x0)
        ## x1:(num_nodes,input_size*batch)

        x = x1.reshape(self._nodes, input_size, -1)
        ## x:(num_nodes,gru_units+1,batch)

        x = x.permute(2,0,1)
        ## x:(batch,num_nodes,gru_units+1)

        x = x.reshape(-1, input_size)
        ## x:(batch * num_nodes,gru_units+1)

        x = self.linear(x)

        x = x.reshape(-1, self._nodes, self._outputsize)
        x = x.reshape(-1, self._nodes * self._outputsize)
        ## x:(batch, num_nodes * output_size)

        return x



class TGCN2(nn.Module):

    def __init__(self, num_nodes, num_features, gru_units, seq_len,
                 pre_len):

        super(TGCN2, self).__init__()

        self._nodes = num_nodes  # 图的节点数
        self._units = gru_units  # gru个数
        self._features = num_features
        self._len = pre_len
        self.gc1 = GCNBlock(num_nodes=num_nodes,gru_units=gru_units,
                            output_size=2*gru_units)
        self.gc2 = GCNBlock(num_nodes=num_nodes, gru_units=gru_units,
                            output_size=gru_units)


        self.linear = nn.Linear(in_features=gru_units,
                                out_features=pre_len)


    def forward(self, A_hat, X, state):
        """
        :param X: Input data of shape (seq_len, batch_size, num_nodes).
        :param state: state of shape (batch_size, num_nodes, gru_units).
        :param A_hat: Normalized adjacency matrix(num_nodes, num_nodes).
        """

        ##hx = torch.zeros(X.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        for i in range(X.size(0)):
            value = torch.sigmoid(self.gc1(A_hat, X[i], state))
            value = value.chunk(2,dim = 1)
            r = value[0]
            u = value[1]
            # r, u: (batch,num_nodes * gru_units)
            r_state = r * state.reshape(len(state),-1)
            c = torch.tanh(self.gc2(A_hat, X[i], r_state.reshape(-1,self._nodes,self._units)))
            state = u * state.reshape(len(state),-1) + (1 - u) * c
            state = state.reshape(-1,self._nodes,self._units)

        ##state: state of shape (batch_size, num_nodes, gru_units).

        output = state.reshape(-1, self._units)
        ## output:(batch * num_nodes,gru_units)

        output = self.linear(output)

        ## output:(batch * num_nodes,pre_len)
        output = output.reshape(-1, self._nodes, self._len)
        ## output:(batch,num_nodes,pre_len)
        output = output.permute(0,2,1)

        return output