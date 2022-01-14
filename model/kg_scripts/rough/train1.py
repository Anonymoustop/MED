import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
from model import transformers_model
# from kg_dataloader import KgDataLoader
import fire
import time
import os
import ast
import fasttext

# uses allennlp modules
from allennlp.nn import util

max_grad_norm = 1.0

# def train_model(

if __name__ == '__main__':
    # print(train_model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
                    help='Choose_model_parameters')
    parser.add_argument('--gpu', default=1, type=int, required=False)
    parser.add_argument('--epochs', default=30, type=int, required=False)
    parser.add_argument('--num_gradients', default=4, type=int, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)
    parser.add_argument('--lr', default=1e-5, type=int, required=False)
    parser.add_argument('--load_dir', default='weights/english/no_kg/med/', type=str, required=False)
    parser.add_argument('--validate_load_dir', default='../../preprocessed_data/data/data_med/validate_data.pkl', type=str, required=False)
    parser.add_argument('--train_load_dir', default='../../preprocessed_data/data/data_med/train_data.pkl', type=str, required=False)
    parser.add_argument('--log_dir', default='log/train.txt', type=str, required=False)
    parser.add_argument('--val_epoch_interval', default=1, type=int, required=False)
    parser.add_argument('--last_epoch_path', default="weights/english/no_kg/med/", type=str, required=False)
    parser.add_argument('--hidden_size', default=512, type=int, required=False)
    parser.add_argument('--vocab_size', default=50000, type=int, required=False)
    parser.add_argument('--finetune', default= 'false', type=str, required=False)
    parser.add_argument('--entity_dic', default='../../triples_prep/covid/entity2id.txt', type=str, required=False)
    parser.add_argument('--rel_dic', default='../../triples_prep/covid/relation2id.txt', type=str, required=False)
    parser.add_argument('--t_embed', default=200, type=int, required=False)

    args = parser.parse_args()

    epochs = args.epochs
    num_gradients_accumulation = args.num_gradients
    batch_size = args.batch_size
    gpu_id = args.gpu
    lr = args.lr
    load_dir = args.load_dir
    validate_load = args.validate_load_dir
    train_load = args.train_load_dir
    log_directory = args.log_dir
    valid_epoch = args.val_epoch_interval

    print(train_load)
    print(validate_load)

    
    save_every = 10
    # ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    f_entity2id = open(args.entity_dic,'r')
    f_rel2id = open(args.rel_dic,'r')

    entity_vocab = int([line for line in f_entity2id][0])
    relation_vocab = int([line for line in f_rel2id][0])

    print(entity_vocab)
    print(relation_vocab)

    # ------------------------LOAD MODEL-----------------
    print('load the model....')
    # print('load the model....')

    f_rel = open('/home1/deeksha/OpenKE/benchmarks/health/relation_vec.txt')

    emb_dim_rel = 200
    embedding_matrix_rel = np.zeros((relation_vocab, emb_dim_rel), dtype=np.float32)

    for line in f_rel:
        index = int(line.split('\t')[0])
        embedding_matrix_rel[index] = ast.literal_eval(line.split('\t')[1].strip())
    emb_weights_rel = torch.from_numpy(embedding_matrix_rel).cuda()

    vocab_dic = {}
    o_vocab_dic = {}

    n_voc = open('../../preprocessing_scripts/kg/biobert/vocab/new_vocab.txt').read().split('\n')[:-1]
    o_voc = open('../../preprocessing_scripts/kg/biobert/vocab/vocab.txt').read().split('\n')[:-1]


    for i,item in enumerate(n_voc):
        vocab_dic[item] = i

    vocab_size = len(vocab_dic)


    for i,item in enumerate(o_voc):
        o_vocab_dic[item] = i

    o_vocab_size = len(o_vocab_dic)
    
    print('vocab_size',vocab_size)

    emb_dim = 300

    embedding_matrix_entity = np.zeros((vocab_size, emb_dim), dtype=np.float32)

    print('loading word embedding...')
    ft = fasttext.load_model("../../../../classifier/wiki.en.bin")

    print('creating embedding matrix...')
    emb_dim = 300
    count = 0
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for i, j in vocab_dic.items():
        word = i
        index = j
        # print(i,j)
        try:
            embedding_vector = ft.get_word_vector(word.lower())
        except:
            # print(word)
            embedding_vector = np.zeros(emb_dim)
            count = count + 1

        embedding_matrix[index] = embedding_vector
        
    print("oov-->",count)
    # emb_weights = torch.from_numpy(embedding_matrix).cuda()

    emb_weights_entity = torch.from_numpy(embedding_matrix).cuda()

    print(emb_weights_entity.size())

    model = transformers_model(args.model_config, args.hidden_size, vocab_size, o_vocab_size, entity_vocab, relation_vocab, args.t_embed, emb_weights_rel, emb_weights_entity)

    # model = transformers_model(args.model_config, args.hidden_size, args.vocab_size)
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)

    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD TRAIN DATA------------------
    train_data = torch.load(train_load)
    #
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)

    val_data = torch.load(validate_load)

    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    # ------------------------END LOAD TRAIN DATA--------------

    # ------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, \
        lr=lr, \
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10, \
        num_training_steps=num_train_optimization_steps
    )
    # ------------------------END SET OPTIMIZER--------------

    # ------------------------START TRAINING-------------------
    update_count = 0

    finetune_check = args.finetune
    PATH = args.last_epoch_path
    if finetune_check == 'false':
        if not os.listdir(PATH):
            print("Training the model from starting")
        else:
            files_int = list()
            for i in os.listdir(PATH):
                if 'best' not in i:
                    epoch = int(i.split('model.')[0])
                    if epoch == 29:
                        files_int.clear()
                        break
                    else:
                        files_int.append(epoch)

            if len(files_int) == 0:
                print("No valid file found from which training should resume")
            else:
                max_value = max(files_int)
                for i in os.listdir(PATH):
                    if 'best' not in i:
                        epoch = int(i.split('model.')[0])
                    if epoch > max_value:
                        pass
                    elif epoch < max_value:
                        pass
                    else:
                        final_file = i
                    print(f'Resuming training from epoch {max_value}')
                    checkpoint = torch.load(PATH + final_file)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']
                    valid_epoch = epoch
                    model.eval()


    if finetune_check == 'true' or finetune_check == 'True':
        print("Initiating finetuning")
        checkpoint = torch.load(PATH + 'bestmodel.pth')
        print(PATH + 'bestmodel.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        model.eval()

    f = open(log_directory, "w")
    f.close()

    start = time.time()
    best_valid_perplexity = float('inf')
    best_valid_epoch = 0
    best_valid_loss = 0
    print('start training....')
    for epoch in range(epochs):
        # ------------------------training------------------------
        f = open(log_directory, "a")
        model.train()
        losses = 0
        times = 0
        print('ee',len(train_dataloader))
        for batch in train_dataloader:
            # print(times)
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, kg_enc_input, entity = batch
            first_logits, second_logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, kg_enc_input, entity)
            # print(first_logits)
            # print(second_logits)
            
            first_out = first_logits[:, :-1].contiguous()

            second_out = second_logits[:, :-1].contiguous()

            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss1 = util.sequence_cross_entropy_with_logits(first_out, target, target_mask, average="token")
            loss2 = util.sequence_cross_entropy_with_logits(second_out, target, target_mask, average="token")
            loss = loss1 + loss2
            # print(loss1)
            # print(loss2)
            
            loss.backward()

            losses += loss.item()

            times += 1
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('-' * 20 + f'epoch {epoch}' + '-' * 20)
        print('-' * 20 + f'epoch {epoch}' + '-' * 20, file=f)
        print(f'time: {(end - start)}')
        print(f'time: {(end - start)}', file=f)
        print(f'loss: {losses / times}')
        print(f'loss: {losses / times}', file=f)
        start = end

        # ------------------------validate------------------------
        # Calculating the perplexity when no of epoch == valid_epoch
        if (epoch + 1) % valid_epoch == 0:
            model.eval()

            perplexity = 0
            batch_count = 0
            print('start calculate the perplexity....')
            print('start calculate the perplexity....', file=f)
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = [item.to(device) for item in batch]

                    encoder_input, decoder_input, mask_encoder_input, mask_decoder_input,  kg_enc_input, entity = batch
                    first_logits, second_logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, kg_enc_input, entity)

                    # first_out = first_logits[:, :-1].contiguous()
                    second_out = second_logits[:, :-1].contiguous()

                    target = decoder_input[:, 1:].contiguous()
                    target_mask = mask_decoder_input[:, 1:].contiguous()

                    # loss1 = util.sequence_cross_entropy_with_logits(first_out, target, target_mask, average="token")
                    loss2 = util.sequence_cross_entropy_with_logits(second_out, target, target_mask, average="token")
                    loss = loss2

                    perplexity += np.exp(loss.item())

                    batch_count += 1

            cur_valid_perplexity = perplexity / batch_count
            cur_valid_loss = loss

            print(f'validate perplexity: {perplexity / batch_count}')
            print(f'validate perplexity: {perplexity / batch_count}', file=f)

            if cur_valid_perplexity < best_valid_perplexity:
                best_valid_perplexity = cur_valid_perplexity
                best_valid_epoch = epoch
                best_valid_loss = cur_valid_loss
                direct_path = os.path.join(os.path.abspath('.'), load_dir)
                if not os.path.exists(direct_path):
                    os.mkdir(direct_path)

                # save_range = range(epochs-5, epochs)
                # if epoch in save_range:
                print(f'saving best model having epoch: {best_valid_epoch}')
                print(f'saving best model having epoch: {best_valid_epoch}', file=f)
                torch.save(
                    {'epoch': best_valid_epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': best_valid_loss}, os.path.join(direct_path, "bestmodel.pth"))

        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)
        # if (epoch + 1) % save_every == 0:
        if epoch == 29:
            print('saving model')
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss}, os.path.join(direct_path, str(epoch) + "model.pth"))

    f.close()

    # ------------------------END TRAINING-------------------