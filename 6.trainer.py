from transformers import AutoTokenizer

if __name__ == '__main__':
    # 第6章/加载tokenizer
    # 首先加载一个编码工具,由于编码工具和模型往往是成对使用的,所以此处使用hfl/rbt3编码工具,因为要再训练的模型是hfl/rbt3模型
    tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
    print(tokenizer)
    # 第6章/试编码句子
    tokenizer.batch_encode_plus(
        ['明月装饰了你的窗子', '你装饰了别人的梦'],
        truncation=True,
    )
    # 第6章/从磁盘加载数据集
    # 加载数据集,使用该数据集来再训练模型,代码如下：
    # 在这段代码中,对数据集进行了采样,目的有以下两方面：一是便于测试；
    # 二是模拟再训练集的体量较小的情况,以验证即使是小的数据集,也能通过迁移学习得到一个较好的训练结果。
    from datasets import load_from_disk

    dataset = load_from_disk('./data/ChnSentiCorp')
    # 缩小数据规模,便于测试
    dataset['train'] = dataset['train'].shuffle().select(range(2000))
    dataset['test'] = dataset['test'].shuffle().select(range(100))
    print(dataset)


    # 第6章/编码
    # 现在的数据集还是文本数据,使用编码工具把这些抽象的文字编码成计算机善于处理的数字,代码如下：
    def f(data):
        tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
        return tokenizer.batch_encode_plus(data['text'], truncation=True)


    # 在这段代码中,使用了批量处理的技巧,能够加快计算的速度。
    # (1)参数batched=True：表明使用批处理来处理数据,而不是一条一条地处理。
    # (2)参数batch_size=1000：表明每个批次中有1000条数据。
    # (3)参数num_proc=4：表明使用4个线程进行操作。
    # (4)参数remove_columns=['text']：表明映射结束后删除数据集中的text字段。
    dataset = dataset.map(f,
                          batched=True,
                          batch_size=1000,
                          num_proc=4,
                          remove_columns=['text'])
    print(dataset)


    # 第6章/移除太长的句子
    # 由于模型对句子的长度有限制,不能处理长度超过512个词的句子,所以需要把数据集中长度超过512个词的句子过滤掉,代码如下：
    def f(data):
        return [len(i) <= 512 for i in data['input_ids']]


    dataset = dataset.filter(f, batched=True, batch_size=1000, num_proc=4)

    print(dataset)
    # 第6章/加载模型
    from transformers import AutoModelForSequenceClassification

    # 如前所述，此处加载的模型应该和编码工具配对使用，所以此处加载的模型为hfl/rbt3模型，
    # 该模型由哈尔滨工业大学讯飞联合实验室(HFL)分享到HuggingFace模型库，这是一个基于中文文本数据训练的BERT模型。
    # 后续将使用准备好的数据集对该模型进行再训练，在代码的最后一行统计了该模型的参数量，以大致衡量一个模型的体量大小。
    # 该模型的参数量约为3800万个，这是一个较小的模型。
    model = AutoModelForSequenceClassification.from_pretrained('hfl/rbt3',
                                                               num_labels=2)

    # 统计模型参数量
    print(sum([i.nelement() for i in model.parameters()]) / 10000)
    # 第6章/模型试算
    import torch

    # 模拟一批数据
    # 加载了模型之后，不妨对模型进行一次试算，以观察模型的输出，代码如下：
    data = {
        'input_ids': torch.ones(4, 10, dtype=torch.long),
        'token_type_ids': torch.ones(4, 10, dtype=torch.long),
        'attention_mask': torch.ones(4, 10, dtype=torch.long),
        'labels': torch.ones(4, dtype=torch.long)
    }

    # 模型试算
    out = model(**data)

    print(out['loss'], out['logits'].shape)
    # 第6章/定义评价函数
    # 为了便于在训练过程中观察模型的性能变化，需要定义一个评价指标函数。
    # 对于情感分类任务往往关注正确率指标，所以此处加载正确率评价函数
    import numpy as np0
    from transformers.trainer_utils import EvalPrediction
    #由于模型计算的输出和评价指标要求的输入还有差别，所以需要定义一个转换函数，
    # 把模型计算的输出转换成评价指标可以计算的数据类型，这个函数就是在训练过程中真正要用到的评价函数，代码如下：

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.argmax(axis=1)
        return {'accuracy': (logits == labels).sum() / len(labels)}
        # return metric.compute(predictions=logits, references=labels)


    # 模拟输出
    eval_pred = EvalPrediction(
        predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
        label_ids=np.array([1, 1, 0, 1]),
    )

    compute_metrics(eval_pred)
    # 第6章/定义训练参数
    from transformers import TrainingArguments

    # 定义训练参数
    args = TrainingArguments(
        # 定义临时数据保存路径
        output_dir='./output_dir',
        # 定义测试执行的策略,可取值no、epoch、steps
        evaluation_strategy='steps',
        # 定义每隔多少个step执行一次测试
        eval_steps=30,
        # 定义模型保存策略,可取值no、epoch、steps
        save_strategy='steps',
        # 定义每隔多少个step保存一次
        save_steps=30,
        # 定义共训练几个轮次
        num_train_epochs=1,
        # 定义学习率
        learning_rate=1e-4,
        # 加入参数权重衰减,防止过拟合
        weight_decay=1e-2,
        # 定义测试和训练时的批次大小
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        # 定义是否要使用gpu训练
        no_cuda=False,
    )
    # 第6章/定义训练器
    from transformers import Trainer
    from transformers.data.data_collator import DataCollatorWithPadding

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    # 第6章/测试数据整理函数
    data_collator = DataCollatorWithPadding(tokenizer)

    # 获取一批数据
    data = dataset['train'][:5]

    # 输出这些句子的长度
    for i in data['input_ids']:
        print(len(i))

    # 调用数据整理函数
    data = data_collator(data)

    # 查看整理后的数据
    for k, v in data.items():
        print(k, v.shape)

    # %%
    tokenizer.decode(data['input_ids'][0])
    # %%
    # 第6章/评价模型
    # trainer.evaluate()
    # %%
    # 第6章/训练
    # trainer.train()
    # %%
    # 第6章/从某个存档继续训练
    # trainer.train(resume_from_checkpoint='./output_dir/checkpoint-90')
    # %%
    # 第6章/评价模型
    # trainer.evaluate()
    # %%
    # 第6章/手动保存模型参数
    # trainer.save_model(output_dir='./output_dir/save_model')
    # %%
    # 第6章/手动加载模型参数
    import torch

    model.load_state_dict(torch.load('./output_dir/save_model/pytorch_model.bin'))
    # 第6章/测试
    model.eval()
    for i, data in enumerate(trainer.get_eval_dataloader()):
        break
    for k, v in data.items():
        # data[k] = v.to('cuda')
        data[k] = v.to()
    out = model(**data)
    out = out['logits'].argmax(dim=1)
    for i in range(16):
        print(tokenizer.decode(data['input_ids'][i], skip_special_tokens=True))
        print('label=', data['labels'][i].item())
        print('predict=', out[i].item())
