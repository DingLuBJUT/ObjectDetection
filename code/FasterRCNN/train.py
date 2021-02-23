from torchvision.models import resnet50


backbone = resnet50(pretrained=True)
# print(backbone)
# print()
# print()
# print(backbone.fc)
# print(backbone.avgpool)
# print(backbone.layer4)

print(backbone.fc.in_features)

# for name,param in backbone.layer4.named_parameters():
#     print(name)
#
# print("*****************")
# print()
#
# for name,modules in backbone.layer4.named_modules():
#     print(name,modules)
#
# print("*****************")
# print()

# for name, children in backbone.named_children():
#     print(name,"********",children.modules())