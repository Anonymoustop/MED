import torch.nn as nn
import json
import argparse
from transformers import BertModel, BertConfig
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
torch.set_printoptions(profile="full")
import torch.nn.functional as F
from multi_headed_attn import MultiHeadedAttention

class Attention(nn.Module):
    def __init__(self, hidden_size, entity_vocab, relation_vocab, t_embed, embedding_matrix_rel, embedding_matrix_entity):
        super().__init__()
        self.t_embed = t_embed
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.hidden_size = hidden_size

        self.entity_embedding = nn.Embedding(self.entity_vocab, 300)
        self.entity_embedding.weight = nn.Parameter(embedding_matrix_entity, requires_grad=True)

        self.rel_embedding = nn.Embedding(self.relation_vocab, self.t_embed)
        self.rel_embedding.weight = nn.Parameter(embedding_matrix_rel, requires_grad=True)
        self.Wh = nn.Linear(self.t_embed, self.hidden_size)
        self.Wr = nn.Linear(self.t_embed, self.hidden_size)
        self.Wt = nn.Linear(self.t_embed, self.hidden_size)
        self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
        self.l = nn.Linear(300, self.t_embed, bias=False)
        self.lii = nn.Linear(2*self.t_embed, 2*self.hidden_size, bias=False)




    def forward(self, kg_enc_input):
        batch_size, _, _, _ = kg_enc_input.size()
        head, rel, tail = torch.split(kg_enc_input, 1, 3)  # (bsz, pl, tl)
        head_emb = self.l(self.entity_embedding(head.squeeze(-1)))  # (bsz, pl, tl, t_embed)
        rel_emb = self.rel_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
        tail_emb = self.l(self.entity_embedding(tail.squeeze(-1)))  # (bsz, pl, tl, t_embed)
        triple_emb = self.MLP(torch.cat([head_emb, rel_emb, tail_emb], 3))  # (bsz, pl, tl, 3 * t_embed)


        ent = torch.cat([head_emb, tail_emb], -1)  # (bsz, pl, tl, 2 * t_embed)
        # mask = get_pad_mask(post_triple.max(-1)[0], ent.size(1)).to(device)
        # ent.data.masked_fill_(mask.view(*mask.size(), 1, 1), 0)
        static_logit = (self.Wr(rel_emb) * torch.tanh(self.Wh(head_emb) + self.Wt(tail_emb))).sum(-1, keepdim=False)  # (bsz, pl, tl)
        # static_logit.data.masked_fill_(triple_mask[:, :, :, 0], -float('inf'))
        # static_logit.data.masked_fill_(post_mask.unsqueeze(-1), 0)
        static_attn = F.softmax(static_logit, dim=-1)  # (bsz, pl, tl) # TODO: NAN
        static_graph = self.lii((ent * static_attn.unsqueeze(-1)).sum(-2))  # (bsz, pl, 2 * t_embed) / gi

        return triple_emb, static_graph

class Pointer(nn.Module):
    def __init__(self, hidden_size, vocab_size, relation_vocab, t_embed, embedding_matrix_rel, embedding_matrix_entity):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.t_embed = t_embed
        self.entity_vocab = vocab_size
        self.relation_vocab = relation_vocab
        self.multi_head = MultiHeadedAttention(8, self.hidden_size, dropout=0.1)
        
        self.attention = Attention(self.hidden_size, self.vocab_size, self.relation_vocab, self.t_embed, embedding_matrix_rel, embedding_matrix_entity)       
        self.linear = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.li = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)

        self.lin = nn.Linear(3 * self.t_embed, self.hidden_size, bias=False)

        self.gate_layer = nn.Sequential(
            nn.Linear(self.vocab_size, 1, bias=True),
            nn.Sigmoid()
        )

        self.copy_gate_layer_kbt = nn.Sequential(
            nn.Linear(self.hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, input_ids, entity, kg_enc_input, out):
        batch_size, _, _, _ = kg_enc_input.size()

        triple_emb,_ = self.attention(kg_enc_input)
        triple_emb = self.lin(triple_emb.view(batch_size,-1,triple_emb.size(-1)))

        # print(out.cross_attentions[1][:,-1:,:,:].size())
        # print(torch.mean(out.cross_attentions[1],1).size())

        dlg_attn = torch.mean(out.cross_attentions[1],1)

        out = out.last_hidden_state
        # print(out.size())
        out = self.li(out)
        
        # print(triple_emb.size())
        mid, attn = self.multi_head(triple_emb, triple_emb, out,
                                        type="knl")
        out = self.linear(out)
        batch_size, max_len, word_size = out.size()

        copy_index = input_ids.unsqueeze(1).expand_as(dlg_attn).contiguous().view(batch_size, max_len, -1)
        copy_logits = dlg_attn.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        copy_logits = copy_logits.scatter_add(dim=2, index=copy_index, src=dlg_attn)
        
        p_gen = self.gate_layer(out)

        # index = entity.view(batch_size,-1).unsqueeze(1).expand_as(attn).contiguous().view(batch_size, max_len, -1)
        # kbt_logits = attn.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        # kbt_logits = kbt_logits.scatter_add(dim=2, index=index, src=attn)
        
        # p_con = self.copy_gate_layer_kbt(mid)

        con_logits = p_gen * out + (1 - p_gen) * copy_logits
        # kb_logits = p_con * kbt_logits + (1 - p_con) * con_logits

        return con_logits

class transformers_model(nn.Module):
    # def __init__(self, config, hidden_size, vocab_size):
    def __init__(self, config, hidden_size, vocab_size, o_vocab_size, entity_vocab, relation_vocab, t_embed, embedding_matrix_rel, embedding_matrix_entity):
        super().__init__()
        # args = setup_train_args()
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.o_vocab_size = o_vocab_size

        self.t_embed = t_embed
        self.entity_vocab = vocab_size
        self.relation_vocab = relation_vocab
        encoder_config = AutoConfig.from_pretrained("./weights/biobert_weight/dmis_biobert_large_case/config.json")

        self.encoder = AutoModel.from_pretrained('./weights/biobert_weight/dmis_biobert_large_case/', config=encoder_config)
            
        decoder_config = AutoConfig.from_pretrained("./weights/biobert_weight/dmis_biobert_large_case/config.json")
        
        decoder_config.is_decoder = True
        
        print(decoder_config)
        encoder_config.add_cross_attention = True
        decoder_config.add_cross_attention = True
        self.attention = Attention(self.hidden_size, self.vocab_size, self.relation_vocab, self.t_embed, embedding_matrix_rel, embedding_matrix_entity)       



        self.li = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)


        self.decoder = AutoModel.from_pretrained('./weights/biobert_weight/dmis_biobert_large_case/', config=decoder_config)
        self.decoder2 = AutoModel.from_pretrained('./weights/biobert_weight/dmis_biobert_large_case/', config=decoder_config)

        # self.linear = nn.Linear(4*self.hidden_size, 2*self.hidden_size, bias=False)

        self.linear = nn.Linear(2*self.hidden_size, self.o_vocab_size, bias=False)
        self.linear2 = nn.Linear(2*self.hidden_size, self.vocab_size, bias=False)

        

        # self.pointer = Pointer(self.hidden_size, self.vocab_size, relation_vocab, t_embed, embedding_matrix_rel, embedding_matrix_entity)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input, kg_enc_input, entity):
        # print(input_ids)
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        triple_emb, static_graph = self.attention(kg_enc_input)
        # print(encoder_hidden_states.size())
        # print(static_graph.size())
        # encoder_hidden_states = self.linear(torch.cat((encoder_hidden_states, static_graph), dim=2))

        out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)

        out = out.last_hidden_state

        first_logits = self.linear(out)
        # print(first_logits.size())
        # print(first_logits)


        _, first_dec_words = torch.max(first_logits, 2)
        # print(first_dec_words.size())
        # print(first_dec_words)


        second_out = self.decoder2(first_dec_words, mask_decoder_input, encoder_hidden_states=static_graph)

        second_logits = self.linear2(second_out.last_hidden_state)
        # print('hh',logits.size())
        # logits = self.pointer(input_ids, entity, kg_enc_input, out)

        return first_logits, second_logits

