
import random

SNs = ['SN001','SN002','SN003','SN004','SN005','SN006','SN007','SN008','SN009',
       'SN010','SN011','SN012','SN013','SN016','SN017','SN018','SN021','SN023',
       'SN024','SN025','SN026','SN027','SN028','SN029','SN030','SN031','SN032']

data_list = ["datasets/DISFA/ALL_FRAMES/" + SN + "/face_only_aligned" for SN in SNs]*450
# data_list = ["datasets/DISFA/ALL_FRAMES/" + SN + "/face_mtcnn" for SN in SNs]*450
random.shuffle(data_list)

# with open("../FaceCycle_with_Modifications/dataloader/DISFA_ALL.txt", 'w') as f:
#     f.write("\n".join(map(str, data_list)))

with open("../FaceCycle_with_Modifications/dataloader/DISFA_ALL_ALL_FRAMES.txt", 'w') as f:
    f.write("\n".join(map(str, data_list)))