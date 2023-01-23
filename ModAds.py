
import torch


class ModAds:
    def __init__(self, labeled_node_dict, unlabeled_node_set, weight_dict, **kwargs):
        kw_dict = {
            'mu_inj': 1,
            'mu_abdn': 1e-4,
            'mu_cont': 1e-4,
            'beta': 2,
            'nn': None,
            'dtype': torch.float32,
            'device': 'cpu'
        }

        self.node_indices, self.unlabeled_index, self.node_len = [], 0, 0
        node_labels, node_index_dict = self.__init_nodes__(labeled_node_dict, unlabeled_node_set)

        assert set(kwargs.keys()) <= set(kw_dict.keys())
        assert min(len(x) for x in node_labels) == max(len(x) for x in node_labels)
        assert 'nn' not in kwargs.keys() or kwargs['nn'] is None or set(kwargs['nn'].keys()) <= {'k', 'bin'}

        kw_dict.update(kwargs)

        with torch.no_grad():
            beta = torch.tensor([kw_dict['beta']], dtype=kw_dict['dtype'])
            beta.to(kw_dict['device'])
            self.mu_cont, self.mu_abdn, self.mu_inj = kw_dict['mu_cont'], kw_dict['mu_abdn'], kw_dict['mu_inj']
            self.W = self.__init_W__(weight_dict, node_index_dict, kw_dict['nn'], kw_dict['dtype'], kw_dict['device'])

            self.Y = torch.tensor(node_labels, dtype=kw_dict['dtype'])
            self.Y.to(kw_dict['device'])
            self.Y_hat = torch.clone(self.Y)

            self.r = torch.tensor([0 for _ in range(self.Y.shape[1] - 1)] + [1], dtype=kw_dict['dtype'])
            self.r.to(kw_dict['device'])

            p_cont = [0 for _ in range(self.node_len)]
            p_inj = [0 for _ in range(self.node_len)]
            p_abdn = [0 for _ in range(self.node_len)]

            for i in range(self.unlabeled_index):
                h_v = torch.dot(self.W[i], torch.log(self.W[i])) * -1
                c_v = (torch.log(beta) / torch.log(beta + torch.exp(h_v))).item()
                d_v = (1 - c_v) * (h_v.item() ** 0.5)
                z_v = max(c_v + d_v, 1)
                p_cont[i], p_inj[i] = c_v / z_v, d_v / z_v
                p_abdn[i] = 1 - (p_cont[i] + p_inj[i])

            for i in range(self.unlabeled_index, self.node_len):
                h_v = torch.dot(self.W[i], torch.log(self.W[i])) * -1
                p_cont[i] = min((torch.log(beta) / torch.log(beta + torch.exp(h_v))).item(), 1)
                p_abdn[i] = 1 - p_cont[i]

            self.p_cont = torch.tensor(p_cont, dtype=kw_dict['dtype'])
            self.p_inj = torch.tensor(p_inj, dtype=kw_dict['dtype'])
            self.p_abdn = torch.tensor(p_abdn, dtype=kw_dict['dtype'])
            self.p_cont.to(kw_dict['device'])
            self.p_inj.to(kw_dict['device'])
            self.p_abdn.to(kw_dict['device'])

            self.M = [0 for _ in range(self.node_len)]

            for i in range(self.node_len):
                main_sum = torch.sum((p_cont[i] * self.W[i]) + (self.p_cont * torch.flatten(self.W[:, i])))
                main_sum -= 2 * p_cont[i] * self.W[i][i]
                self.M[i] = 1 / ((self.mu_inj * p_inj[i]) + (self.mu_cont * main_sum.item()) + self.mu_abdn)

    def __init_W__(self, weight_dict, node_dict, nn, dtype, device):
        temp_w = torch.zeros((self.node_len, self.node_len), dtype=dtype)

        for a, b in weight_dict.keys():
            a_i, b_i = node_dict[a], node_dict[b]
            temp_w[a_i, b_i] = weight_dict[(a, b)]

        if nn is None:
            out_w = temp_w
            out_w.to('cpu')
            out_w = out_w.type(torch.float32)
        else:
            k_range = range(nn['k'] if 'k' in nn.keys() else min(5, self.node_len))
            bin_vals = nn['bin'] if 'bin' in nn.keys() else False

            assert self.node_len not in set(k_range) and type(bin_vals) is bool

            out_w = torch.zeros((self.node_len, self.node_len), dtype=torch.float32)
            out_w.to('cpu')

            for i in range(self.node_len):
                w_i = temp_w[i]
                w_i[i] = 0

                for _ in k_range:
                    argmax = torch.argmax(w_i).item()
                    argmax_val = w_i[argmax]
                    out_w[i, argmax] = argmax_val
                    w_i[argmax] = 0

                out_w[i, i] = 1

            if bin_vals:
                out_w.apply_(lambda x: 1 if x > 0 else 0)

        out_w.apply_(lambda x: max(x, 1e-10))
        out_w.to(device)
        out_w = out_w.type(dtype)

        return out_w

    def __init_nodes__(self, labeled_node_dict, unlabeled_node_set):
        self.node_indices = list(labeled_node_dict.keys())
        self.unlabeled_index = len(self.node_indices)
        self.node_indices += list(unlabeled_node_set)
        self.node_len = len(self.node_indices)
        node_labels = [labeled_node_dict[self.node_indices[i]] + [0] for i in range(self.unlabeled_index)]
        label_range = range(len(node_labels[0]))
        node_labels += [[0 for _ in label_range] for _ in range(self.unlabeled_index, self.node_len)]

        return node_labels, {self.node_indices[i]: i for i in range(len(self.node_indices))}

    def run_modads(self, epsilon=1e-3):
        with torch.no_grad():
            delta = epsilon + 10000000
            prev_y_hat = torch.clone(self.Y_hat)

            while delta > epsilon:
                for i in range(self.node_len):
                    cont_t = (self.p_cont[i] * self.W[i]) + (self.p_cont * torch.flatten(self.W[:, i]))
                    d_v = torch.sum(cont_t[:, None] * self.Y_hat, dim=0)
                    self.Y_hat[i] = self.M[i] * ((self.mu_inj * self.p_inj[i] * self.Y[i]) + (self.mu_cont * d_v) + (self.p_abdn[i] * self.r))

                delta = torch.sum(torch.abs(prev_y_hat - self.Y_hat)).item()
                prev_y_hat = torch.clone(self.Y_hat)

            y_hat_list = self.Y_hat.tolist()

            return {
                'LABELED': {self.node_indices[i]: y_hat_list[i][:-1] for i in range(self.unlabeled_index)},
                'UNLABELED': {self.node_indices[i]: y_hat_list[i][:-1] for i in range(self.unlabeled_index, self.node_len)}
            }


LABELED_NODES = {
    'A': [2, 1, 1],
    'B': [0, 5, 0],
    'C': [1, 0, 1]
}
UNLABELED_NODES = {'D', 'E'}
WEIGHTS = {
    ('A', 'B'): 0.5,
    ('B', 'A'): 0.3,
    ('B', 'C'): 0.11,
    ('D', 'C'): 0.9,
    ('D', 'E'): 0.53,
    ('C', 'D'): 0.75
}

modads = ModAds(LABELED_NODES, UNLABELED_NODES, WEIGHTS)
out_dict = modads.run_modads()
