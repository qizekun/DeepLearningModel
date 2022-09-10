import time
import torch
from utils.tools import AccuracyMeter, TenCropsTest, data_analysis


def train_one_iter(net, train_iter, device, loss_function, optimizer):
    net.train()
    train_inputs, train_labels = next(train_iter)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    train_outputs = net(train_inputs)
    loss = loss_function(train_outputs, train_labels)

    net.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def val_one_iter(net, val_loader, device, loss_function):
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
    iter_nums = train_len * epochs
    running_loss = step = epoch = 0
    t = time.perf_counter()
    train_iter = iter(train_loader)
    for iter_num in range(iter_nums):

        loss = train_one_iter(net, train_iter, device, loss_function, optimizer)
        running_loss += loss
        step += 1
        rate = step / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtraining: {:^3.0f}%[{}->{}]".format(int(rate * 101), a, b), end="")

        if iter_num % train_len == 0 and iter_num != 0:
            train_iter = iter(train_loader)
            epoch += 1
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            t = round(time.perf_counter() - t, 3)

            val_loss, val_accurate = val_one_iter(net, val_loader, device, loss_function)
            if val_accurate > train_info['best_acc']:
                train_info['best_acc'] = float(val_accurate)
                train_info['best_epoch'] = epoch
                if save:
                    torch.save(net.state_dict(), save_path)

            print('\ntime:%e' % t, f'\tlr:{lr}', f'\tbest_acc:{round(train_info["best_acc"], 3)}')

            scheduler.step()  # 动态调整学习率
            train_info['all_val_accurate'].append(val_accurate / 100)
            train_info['all_train_loss'].append(running_loss / len(train_loader))
            train_info['all_val_loss'].append(val_loss / len(val_loader))
            print('[epoch %d] train_loss: %.3f val_loss: %.3f  test_accuracy: %.3f' %
                  (epoch, running_loss / len(train_loader), val_loss / len(val_loader), val_accurate))
            train_info['epoch'] = epoch
            data_analysis(model)
            running_loss = step = 0
            t = time.perf_counter()
    if test:
        test_acc = TenCropsTest(test_loaders, net)
        print(f'Finished Training! Test acc: {test_acc}')
