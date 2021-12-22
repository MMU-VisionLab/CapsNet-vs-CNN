'''
Script to test a single model on the dataset both on feature-level and object-level.
'''

from tqdm import tqdm
import cv2
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from runtime_args import args
from model_factory.model_factory import ModelFactory
from load_dataset import LoadDataset
import helper


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")


########################################### FEATURE LEVEL EVALUATION ###########################################
feature_level_train_data = LoadDataset(dataset_folder_path='./generated_dataset', level='feature_level', train=True, transform=transforms.ToTensor())
feature_level_test_data = LoadDataset(dataset_folder_path='./generated_dataset', level='feature_level', train=False, transform=transforms.ToTensor())

#if any of the arguments for the model initialization needs to be changed, it can be passed as a parameter here. The factory will override the default
#parameters for the models set in the model configuration files.
model, criterion, optimizer, lr_decayer = ModelFactory().factory_call(requested_model=args.model, mode=None,
                                                                      model_load_path=None, device=DEVICE, image_size=args.img_size,
                                                                      learning_rate=args.learning_rate)


feat_train_generator = DataLoader(feature_level_train_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
feat_test_generator = DataLoader(feature_level_test_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

if not helper.check_dir_exists(args.model_save_path) : helper.create_dir(args.model_save_path)

model = model.to(DEVICE)

summary(model, (1, args.img_size, args.img_size))

if args.model == 'simple_cnn':

    print("Feature-level evaluation has for simple CNN has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_feature):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_cnn_model(generator=feat_train_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)


        model.eval()
        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_cnn_model(generator=feat_test_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)


        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        #save the model when it has the best accuracy.
        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/simple_cnn_feature_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Simple CNN feature-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


elif args.model == 'simple_capsnet':

    print("Feature-level evaluation for simple CapsNet has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_feature):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_capsnet_model(generator=feat_train_generator, model=model, criterion=criterion,
                                                                        optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)

        model.eval()

        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_capsnet_model(generator=feat_test_generator, model=model, criterion=criterion,
                                                                        optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)


        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/simple_capsnet_feature_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Simple CapsNet feature-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


elif args.model == 'deep_capsnet':

    print("Feature-level evaluation for Deep CapsNet has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_feature):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_deepcaps_model(epoch_idx=epoch_idx,
                                                                                        generator=feat_train_generator, model=model, criterion=criterion,
                                                                                        optimizer=optimizer, lr_decayer=lr_decayer, num_classes=args.num_classes,
                                                                                        device=DEVICE, train=True)

        model.eval()

        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_deepcaps_model(epoch_idx=epoch_idx,
                                                                                    generator=feat_test_generator, model=model, criterion=criterion,
                                                                                    optimizer=optimizer, lr_decayer=lr_decayer, num_classes=args.num_classes,
                                                                                    device=DEVICE, train=False)

        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/deep_capsnet_feature_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Deep CapsNet feature-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


elif args.model == 'deep_cnn':

    print("Feature-level evaluation has for Deep CNN has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_feature):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_cnn_model(generator=feat_train_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)


        model.eval()
        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_cnn_model(generator=feat_test_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)


        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        #save the model when it has the best accuracy.
        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/deep_cnn_feature_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Deep CNN feature-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


########################################### OBJECT LEVEL EVALUATION ###########################################
object_level_train_data = LoadDataset(dataset_folder_path='./generated_dataset', level='object_level', train=True, transform=transforms.ToTensor())
object_level_test_data = LoadDataset(dataset_folder_path='./generated_dataset', level='object_level', train=False, transform=transforms.ToTensor())

#if any of the arguments for the model initialization needs to be changed, it can be passed as a parameter here. The factory will override the default
#parameters for the models set in the model configuration files.
model, criterion, optimizer, lr_decayer = ModelFactory().factory_call(requested_model=args.model, mode='transfer_learning',
                                                                      model_load_path=args.model_save_path,
                                                                      device=DEVICE, image_size=args.img_size)


obj_train_generator = DataLoader(object_level_train_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
obj_test_generator = DataLoader(object_level_test_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

model = model.to(DEVICE)

summary(model, (1, args.img_size, args.img_size))

if args.model == 'simple_cnn':

    print("Object-level evaluation has for simple CNN has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_object):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_cnn_model(generator=obj_train_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)


        model.eval()
        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_cnn_model(generator=obj_test_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)



        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        #save the model when it has the best accuracy.
        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/simple_cnn_object_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")


    print("The best train and test accuracy for Simple CNN object-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


elif args.model == 'simple_capsnet':

    print("Object-level evaluation for simple CapsNet has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_object):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_capsnet_model(generator=obj_train_generator, model=model, criterion=criterion,
                                                                        optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)

        model.eval()

        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_capsnet_model(generator=obj_test_generator, model=model, criterion=criterion,
                                                                        optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)


        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/simple_capsnet_object_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Simple CapsNet object-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))


elif args.model == 'deep_capsnet':

    print("Object-level evaluation for Deep CapsNet has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_object):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_deepcaps_model(epoch_idx=epoch_idx, generator=obj_train_generator, model=model,
                                                                                        criterion=criterion,
                                                                                        optimizer=optimizer, lr_decayer=lr_decayer, num_classes=args.num_classes,
                                                                                        device=DEVICE, train=True)

        model.eval()

        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_deepcaps_model(epoch_idx=epoch_idx, generator=obj_test_generator, model=model,
                                                                                    criterion=criterion,
                                                                                    optimizer=optimizer, lr_decayer=lr_decayer, num_classes=args.num_classes,
                                                                                    device=DEVICE, train=False)

        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/deep_capsnet_object_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Deep CapsNet object-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))



elif args.model == 'deep_cnn':

    print("Object-level evaluation has for Deep CNN has started!")

    best_accuracy = [0,0]
    for epoch_idx in range(args.epoch_object):

        model.train()

        train_epoch_loss, train_epoch_accuracy, train_runs = helper.run_cnn_model(generator=obj_train_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=True)


        model.eval()
        test_epoch_loss, test_epoch_accuracy, test_runs = helper.run_cnn_model(generator=obj_test_generator, model=model, criterion=criterion,
                                                                  optimizer=optimizer, lr_decayer=lr_decayer, device=DEVICE, train=False)


        print('----------- Epoch '+ str(epoch_idx) + '-----------')
        print(f"Mean Training Loss : {train_epoch_loss/(train_runs + 1)}")
        print(f"Mean Training Accuracy : {train_epoch_accuracy/(train_runs + 1)}")
        print(f"Mean Testing Loss : {test_epoch_loss/(test_runs + 1)}")
        print(f"Mean Testing Accuracy : {test_epoch_accuracy/(test_runs + 1)}")

        #save the model when it has the best accuracy.
        if best_accuracy[1] <= test_epoch_accuracy/(test_runs + 1):
            torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/deep_cnn_object_model.pth')
            best_accuracy = [train_epoch_accuracy/(train_runs + 1), test_epoch_accuracy/(test_runs + 1)]
            print("Model is saved!")

    print("The best train and test accuracy for Deep CNN object-level eval is %g and %g"%(best_accuracy[0], best_accuracy[1]))