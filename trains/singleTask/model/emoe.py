import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from ...singleTask.model.router import router

class EMOE(nn.Module):
    def __init__(self, args):
        super(EMOE, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.aligned = args.need_data_aligned
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.fusion_method = args.fusion_method
        output_dim = 1
        self.args = args

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        self.self_attentions_l = self.get_network(self_type='l')
        self.self_attentions_v = self.get_network(self_type='v')
        self.self_attentions_a = self.get_network(self_type='a')

        self.proj1_l = nn.Linear(self.d_l, self.d_l)
        self.proj2_l = nn.Linear(self.d_l, self.d_l)
        self.out_layer_l = nn.Linear(self.d_l, output_dim)
        self.proj1_v = nn.Linear(self.d_l, self.d_l)
        self.proj2_v = nn.Linear(self.d_l, self.d_l)
        self.out_layer_v = nn.Linear(self.d_l, output_dim)
        self.proj1_a = nn.Linear(self.d_l, self.d_l)
        self.proj2_a = nn.Linear(self.d_l, self.d_l)
        self.out_layer_a = nn.Linear(self.d_l, output_dim)

        if self.fusion_method == "sum": 
            self.proj1_c = nn.Linear(self.d_l, self.d_l)
            self.proj2_c = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c = nn.Linear(self.d_l, output_dim)
        elif self.fusion_method == "concat":
            self.proj1_c = nn.Linear(self.d_l*3, self.d_l*3)
            self.proj2_c = nn.Linear(self.d_l*3, self.d_l*3)
            self.out_layer_c = nn.Linear(self.d_l*3, output_dim)

        self.Router = router(self.orig_d_l * self.len_l + self.orig_d_a * self.len_l + self.orig_d_v * self.len_l, 3, self.args.temperature)
        self.transfer_a_ali = nn.Linear(self.len_a, self.len_l)
        self.transfer_v_ali = nn.Linear(self.len_v, self.len_l)


    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_net(self, name):
        return getattr(self, name)

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        if not self.aligned:
            audio_ = self.transfer_a_ali(audio.permute(0, 2, 1)).permute(0, 2, 1)
            video_ = self.transfer_v_ali(video.permute(0, 2, 1)).permute(0, 2, 1)
            m_i = torch.cat((text, video_, audio_), dim=2)
        else:
            m_i = torch.cat((text, video, audio), dim=2)
        m_w = self.Router(m_i)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        c_l_att = self.self_attentions_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        l_proj = self.proj2_l(
            F.dropout(F.relu(self.proj1_l(c_l_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        l_proj += c_l_att
        logits_l = self.out_layer_l(l_proj)
        v_proj = self.proj2_v(
            F.dropout(F.relu(self.proj1_v(c_v_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        v_proj += c_v_att
        logits_v = self.out_layer_v(v_proj)
        a_proj = self.proj2_a(
            F.dropout(F.relu(self.proj1_a(c_a_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        a_proj += c_a_att
        logits_a = self.out_layer_a(a_proj)

        if self.fusion_method == "sum": 
            for i in range(m_w.shape[0]):
                c_f = c_l_att[i] * m_w[i][0] + c_v_att[i] * m_w[i][1] + c_a_att[i] * m_w[i][2]
                if i == 0: c_fusion = c_f.unsqueeze(0)
                else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   
        elif self.fusion_method == "concat":        
            for i in range(m_w.shape[0]):
                c_f = torch.cat([c_l_att[i] * m_w[i][0], c_v_att[i] * m_w[i][1], c_a_att[i] * m_w[i][2]], dim=0) * 3
                if i == 0: c_fusion = c_f.unsqueeze(0)
                else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   


        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)


        res = {
            'logits_c': logits_c,
            'logits_l': logits_l,
            'logits_v': logits_v,
            'logits_a': logits_a,
            'channel_weight': m_w,
            'c_proj': c_proj,
            'l_proj': l_proj,
            'v_proj': v_proj,
            'a_proj': a_proj,
            'c_fea': c_fusion,
        }
        return res