from reciperec.models.bpr import BPR
from reciperec.models.itemknn import ItemKNN
from reciperec.models.popularity import Popularity
from reciperec.models.puresvd import PureSVD

MODELS = {"popularity": Popularity, "itemknn": ItemKNN, "puresvd": PureSVD, "bpr": BPR}
