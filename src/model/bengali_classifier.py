from model import senet
from torch import nn
import torch
import torch.nn.functional as F

from model.pooling import GeM
from model.pooling import GlobalAvePooling as GAP
from model.pooling import MaskingPooling
from model import mish
from model import multiscale as mlts


class ClassifierModule_v1(nn.Module):
    def __init__(self, encoded_planes, 
                 encoder_is_separately=False,
                 dropout_p=None, gem_p=3,
                 n_grapheme=168, n_vowel=11, n_consonant=7):
        super(ClassifierModule_v1, self).__init__()

        self.encoder_is_separately = encoder_is_separately

        self.encoded_dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        if self.encoder_is_separately:
            if gem_p is not None:
                self.gra_gem = GeM(p=gem_p)
                self.vow_gem = GeM(p=gem_p)
                self.con_gem = GeM(p=gem_p)
            else:
                self.gra_gem = GAP()
                self.vow_gem = GAP()
                self.con_gem = GAP()
        else:
            if gem_p is not None:
                self.gem = GeM(p=gem_p)
            else:
                self.gem = GAP()

        self.gra_module = nn.Linear(encoded_planes, n_grapheme)
        self.vow_module = nn.Linear(encoded_planes, n_vowel)
        self.con_module = nn.Linear(encoded_planes, n_consonant)

    def forward(self, x):
        encoded = x

        if self.encoder_is_separately:
            def _calc_logit(_x, _gem, _logit_module):
                _y = _gem(_x)
                _y = _y.view(_y.size(0), -1)
                if self.encoded_dropout is not None:
                    _y = self.encoded_dropout(_y)
                _y = _logit_module(_y)
                return _y

            gra_logit = _calc_logit(encoded[0], self.gra_gem, self.gra_module)
            vow_logit = _calc_logit(encoded[1], self.vow_gem, self.vow_module)
            con_logit = _calc_logit(encoded[2], self.con_gem, self.con_module)

        else:
            encoded = self.gem(encoded)
            encoded = encoded.view(encoded.size(0), -1)

            if self.encoded_dropout is not None:
                encoded = self.encoded_dropout(encoded)

            gra_logit = self.gra_module(encoded)
            vow_logit = self.vow_module(encoded)
            con_logit = self.con_module(encoded)

        return gra_logit, vow_logit, con_logit

class ClassifierModule_v9(nn.Module):
    """
    SENetEncoder_Multiscale_v2
    """
    def __init__(self, encoded_planes, 
                 encoder_is_separately=False,
                 dropout_p=0.0, gem_p=None, 
                 n_grapheme=168, n_vowel=11, n_consonant=7):
        super(ClassifierModule_v9, self).__init__()

        self.gra_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_grapheme, dropout_p=dropout_p)
        self.vow_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_vowel, dropout_p=dropout_p)
        self.con_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_consonant, dropout_p=dropout_p)

    def forward(self, x):
        encoded = x

        gra_logit = self.gra_module(encoded)
        vow_logit = self.vow_module(encoded)
        con_logit = self.con_module(encoded)

        return gra_logit, vow_logit, con_logit

class ClassifierModule_v10(nn.Module):
    """
    SENetEncoder_Multiscale_v2
    predict unique
    """
    def __init__(self, encoded_planes, 
                 encoder_is_separately=False,
                 dropout_p=0.0, gem_p=None, 
                 n_grapheme=168, n_vowel=11, n_consonant=7, n_unique=1292):
        super(ClassifierModule_v10, self).__init__()

        self.gra_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_grapheme, dropout_p=dropout_p)
        self.vow_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_vowel, dropout_p=dropout_p)
        self.con_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_consonant, dropout_p=dropout_p)

        n_sum = n_unique
        self.uni_module = mlts.Mix_v1(x_lengths=encoded_planes, num_output=n_sum, dropout_p=dropout_p)

    def forward(self, x):
        encoded = x

        gra_logit = self.gra_module(encoded)
        vow_logit = self.vow_module(encoded)
        con_logit = self.con_module(encoded)

        if self.training:
            uni_logit = self.uni_module(encoded)
            return gra_logit, vow_logit, con_logit, uni_logit
        else:
            return gra_logit, vow_logit, con_logit


class BengaliClassifier_v1(nn.Module):
    def __init__(self, encoder, encoded_planes, 
                 encoder_is_separately=False,
                 encoder_use_mixup=False,
                 dropout_p=None, gem_p=3,
                 n_grapheme=168, n_vowel=11, n_consonant=7,
                 classifier_module=ClassifierModule_v1,
                 classifier_use_orthogonality=False,
                 classifier_use_class_orthogonality=False,
                 ):
        super(BengaliClassifier_v1, self).__init__()
        self.encoder_is_separately = encoder_is_separately
        self.encoder_use_mixup = encoder_use_mixup
        self.classifier_use_orthogonality = classifier_use_orthogonality
        self.classifier_use_class_orthogonality = classifier_use_class_orthogonality

        self.encoder = encoder
        self.classifier = classifier_module(encoded_planes, self.encoder_is_separately, dropout_p, gem_p, n_grapheme, n_vowel, n_consonant)

    def freeze_encoder(self, freeze=True, only_first_layer=False, only_attention=False, target_layers=None):
        if only_first_layer:
            self.encoder.freeze_first_layer(freeze)
        elif only_attention:
            self.encoder.freeze_attention(freeze)
        elif target_layers is not None:
            self.encoder.freeze_layer(freeze, target_layers)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = not freeze
        return

    def forward(self, x, label=None):

        if self.training:
            # encod
            if self.encoder_use_mixup:
                encoded, label_a, label_b, mix_rate = self.encoder(x, label)
            else:
                encoded = self.encoder(x, label)

            # classify
            if self.classifier_use_orthogonality:
                if self.classifier_use_class_orthogonality:
                    gra_logit, vow_logit, con_logit, orth = self.classifier(encoded, label_a, label_b, mix_rate)
                else:
                    gra_logit, vow_logit, con_logit, orth = self.classifier(encoded)
            else:
                gra_logit, vow_logit, con_logit = self.classifier(encoded)

            # return
            if self.encoder_use_mixup and self.classifier_use_orthogonality:
                return (gra_logit, vow_logit, con_logit), label_a, label_b, mix_rate, orth
            elif self.encoder_use_mixup:
                return (gra_logit, vow_logit, con_logit), label_a, label_b, mix_rate
            elif self.classifier_use_orthogonality:
                return (gra_logit, vow_logit, con_logit), mix_rate, orth
            else:
                return gra_logit, vow_logit, con_logit

        else:
            # encod
            encoded = self.encoder(x)

            # classify
            gra_logit, vow_logit, con_logit = self.classifier(encoded)

            return gra_logit, vow_logit, con_logit

class BengaliClassifier_v2(nn.Module):
    def __init__(self, encoder, classifier, 
                 encoder_use_mixup=False,
                 ):
        super(BengaliClassifier_v2, self).__init__()
        self.encoder_use_mixup = encoder_use_mixup

        self.encoder = encoder
        self.classifier = classifier

    def freeze_encoder(self, freeze=True, target_layers=None):
        if target_layers is not None:
            self.encoder.freeze_layer(freeze, target_layers)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = not freeze
        return

    def forward(self, x, label=None):

        if self.training:
            # encod
            if self.encoder_use_mixup:
                encoded, label_a, label_b, mix_rate = self.encoder(x, label)
            else:
                encoded = self.encoder(x, label)

            # classify
            outputs = self.classifier(encoded)

            # return
            if self.encoder_use_mixup:
                return outputs, label_a, label_b, mix_rate
            else:
                return outputs

        else:
            # encod
            encoded = self.encoder(x)

            # classify
            outputs = self.classifier(encoded)

            return outputs
