import time
import torch
from utils.tools import AccuracyMeter, TenCropsTest, data_analysis


def train_one_iter(net, train_iter, device, loss_function, optimizer, acc_meter):
    net.train()
    train_inputs, train_labels = next(train_iter)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    train_outputs = net(train_inputs)
    loss = loss_function(train_outputs, train_labels)
    acc_meter.update(train_outputs, train_labels)

    net.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def val_one_epoch(net, val_loader, device, loss_function):
    acc_meter = AccuracyMeter(topk=(1,))
    with torch.no_grad():
        net.eval()
        val_loss = 0.0
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = net(val_inputs)
            val_loss += loss_function(val_outputs, val_labels)
            acc_meter.update(val_outputs, val_labels)
        val_accurate = acc_meter.avg[1]
        acc_meter.reset()
    return val_loss, val_accurate


def train_model(model, train_loader, val_loader, test_loaders, epochs, net, device, loss_function, optimizer,
                scheduler, train_info, save, save_path, test=True):
    train_len = len(train_loader) - 1
    iter_nums = train_len * (epochs + 1)
    running_loss = step = epoch = 0
    t = time.perf_counter()
    acc_meter = AccuracyMeter(topk=(1,))
    train_iter = iter(train_loader)

    for iter_num in range(iter_nums):
        loss = train_one_iter(net, train_iter, device, loss_function, optimizer, acc_meter)
        running_loss += loss
        step += 1
        rate = step / train_len
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r[epoch {}] training: {:^3.0f}%[{}->{}]".format(epoch, int(rate * 100), a, b), end="")

        # 每个epoch结束后进行eval
        if iter_num % train_len == 0 and iter_num != 0:
            train_iter = iter(train_loader)
            train_accurate = acc_meter.avg[1]
            acc_meter.reset()

            val_loss, val_accurate = val_one_epoch(net, val_loader, device, loss_function)
            if val_accurate > train_info['best_acc']:
                train_info['best_acc'] = float(val_accurate)
                train_info['best_epoch'] = epoch
                if save:
                    torch.save(net.state_dict(), save_path)

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            cost_time = round(time.perf_counter() - t, 2)
            best_acc = round(train_info["best_acc"], 2)
            print('\ntime:%.2f\tlr:%.2e\tbest_acc:%.2f' % (cost_time, lr, best_acc))

            # 动态调整学习率
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accurate)
            else:
                scheduler.step()
            train_info['all_val_accurate'].append(val_accurate.cpu().numpy() / 100)
            train_info['all_train_loss'].append(running_loss.detach().cpu().numpy() / len(train_loader))
            train_info['all_val_loss'].append(val_loss.detach().cpu().numpy() / len(val_loader))
            print('[epoch %d] train_loss:%.3f  val_loss:%.3f  train_acc:%.2f  val_acc:%.2f' %
                  (epoch, running_loss / len(train_loader), val_loss / len(val_loader), train_accurate, val_accurate))
            train_info['epoch'] = epoch

            data_analysis(model)
            epoch += 1
            running_loss = step = 0
            t = time.perf_counter()
    if test:
        print('\nStart testing...')
        test_acc = TenCropsTest(test_loaders, net)
        print(f'Finished Training! Test acc: {test_acc}')
