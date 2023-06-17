# This is a sample Python script.
from pathlib import Path
from typing import List, Dict, Union

import numpy as np

import torch
from torch import nn, Tensor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.nn.functional import relu
from scipy.io.wavfile import read as read_wav


class VideoResnetFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoResnetFeatureExtractor, self).__init__()
        video_resnet = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.stem = video_resnet.stem

        self.vresnet_feature_network = nn.Sequential(
            video_resnet.layer1, video_resnet.layer2
        )
        self.ste = SpatioTemporalEncoder()

    def forward(self, x: Tensor):
        # x has a shape of [B, C, D, H, W]
        # D = 32

        x = x.permute(0,2,1,3,4)

        x = self.stem(x)
        x = self.vresnet_feature_network(x)
        ste_features = self.ste(x)
        x = x.permute(0,2,3,4,1)
        x = torch.cat([x,ste_features],dim=-1)
        return x

class AudioProcessingNetwork(nn.Module):
    def __init__(self, requires_grad: bool, sample_rate: int):
        assert (
                sample_rate is not None
        ), "sample rate of the audio source is required for the network to proces it."

        super(AudioProcessingNetwork, self).__init__()
        self.sample_rate = sample_rate
        self.VGGish: nn.Module = torch.hub.load("harritaylor/torchvggish", "vggish")
        for param in self.VGGish.parameters():
            param.requires_grad = requires_grad
        self.fc = nn.Linear(128 * 3, 100)

    def forward(self, audio_signal_batch: List[np.ndarray]):
        # TODO batch processing of audio chunks?
        batch_size = len(audio_signal_batch)
        outs = [self.VGGish.forward(audio_signal, self.sample_rate) for audio_signal in audio_signal_batch]
        out = torch.stack(outs,dim=0)
        out = out.reshape(batch_size,-1)
        out = self.fc(out)
        out = out.reshape(batch_size, 1, 1, 1, 100)
        return out

class QueryPreprocessor(nn.Module):
    def __init__(self, embed_dim: int):
        super(QueryPreprocessor, self).__init__()
        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3d = nn.Conv3d(in_channels=148, out_channels=16, kernel_size=1)
        self.activation = nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc = nn.Linear(in_features=6272, out_features=embed_dim)

    def forward(self, x: Tensor):
        x = x.permute(0,4,1,2,3)
        #(N, C, D, H, W)

        x = self.maxpool3d(x)
        x = self.activation(self.conv3d(x))
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.maxpool2d(x)
        x = self.activation(self.conv2d(x))
        x = x.reshape(x.shape[0], 1, -1)
        x = self.activation(self.fc(x))
        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(self):
        super(SpatioTemporalEncoder, self).__init__()
        self.temp_enc_t1 = nn.Linear(1, 20)
        self.temp_enc_t2 = nn.Linear(20, 10)

        self.spat_enc_t1 = nn.Linear(2, 20)
        self.spat_enc_t2 = nn.Linear(20, 10)

        self.encs_final_shape = (16, 28, 28,10)

    def forward(self,x: Tensor):
        batch_size = x.shape[0]
        encs_final_shape = [batch_size, *self.encs_final_shape]
        spat_enc = (
            torch.tensor([[i - 14, j - 14] for i in range(28) for j in range(28) for b in range(batch_size)])
            .view(28 * 28, 2)
            .to(device=x.device,dtype=torch.float32)
        )
        temp_enc = torch.arange(-8, 8, 1).view(16, 1).to(torch.float32)
        temp_enc = torch.stack(batch_size*[temp_enc])

        temp_enc = nn.functional.relu(self.temp_enc_t1(temp_enc))
        temp_enc = nn.functional.relu(self.temp_enc_t2(temp_enc))

        temp_enc = temp_enc[:,:,None,None,:].expand(encs_final_shape)

        spat_enc = nn.functional.relu(self.spat_enc_t1(spat_enc))
        spat_enc = nn.functional.relu(self.spat_enc_t2(spat_enc))
        spat_enc = spat_enc.reshape(batch_size,1, 28, 28, 10).expand(encs_final_shape)

        encodings = torch.concat([spat_enc, temp_enc], dim=-1)

        return encodings


class TxUnit(nn.Module):
    def __init__(self, embed_dim: int, key_embed_dim: int, value_embed_dim: int):
        super(
            TxUnit,
            self,
        ).__init__()
        self.embed_dim = embed_dim
        self.key_embed_dim = key_embed_dim
        self.value_embed_dim = value_embed_dim
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, kdim=key_embed_dim, vdim=value_embed_dim,
                                         batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.layernorm1 = nn.LayerNorm([1, 128])
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.layernorm2 = nn.LayerNorm([1, 128])

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        q = query.reshape(-1, self.embed_dim)
        k = key.reshape(-1, self.key_embed_dim)
        v = value.reshape(-1, self.value_embed_dim)

        mha_out = self.MHA.forward(q, k, v)

        d1_out = self.dropout1(mha_out[0])
        add1_out = d1_out + q
        layernorm1_out = self.layernorm1(add1_out)
        fc1_out = nn.functional.relu(self.fc1(layernorm1_out))
        fc2_out = nn.functional.relu(self.fc2(fc1_out))
        d2_out = self.dropout2(fc2_out)
        add2_out = layernorm1_out + layernorm1_out
        layernorm2_out = self.layernorm2(add2_out)
        return layernorm2_out


class TxKeyValPreprocessor(nn.Module):
    def __init__(self, embed_dim: int, k_input_dim: int, v_input_dim: int):
        super(TxKeyValPreprocessor, self).__init__()
        self.fck = nn.Linear(k_input_dim, embed_dim)
        self.fcv = nn.Linear(v_input_dim, embed_dim)

    def forward(self, features: Tensor):

        key = relu(self.fck(features))
        value = relu(self.fcv(features))
        return (key, value)


class TxQueryPreprocessor(nn.Module):
    def __init__(self, in_features: int, embed_dim: int):
        super(TxQueryPreprocessor, self).__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, query_features, metadata_features):
        out = torch.concat([query_features, metadata_features], dim=-1)
        out = nn.functional.relu(self.fc(out))
        return out


class TxLayer(nn.Module):
    def __init__(self, num_units: int, embed_dim: int, key_embed_dim: int, value_embed_dim: int):
        super(TxLayer, self).__init__()

        self.tx_units: List[TxUnit] = [
            TxUnit(embed_dim=embed_dim, key_embed_dim=key_embed_dim, value_embed_dim=value_embed_dim) for _ in
            range(0, num_units)]
        self.fc = nn.Linear(num_units * embed_dim, embed_dim)

    def forward(self, query: Tensor, keys: List[Tensor], values: List[Tensor]):
        updated_queries: List[Tensor] = [tx_unit.forward(query, key, value) for tx_unit, key, value in
                                         zip(self.tx_units, keys, values)]
        updated_queries_ten: Tensor = torch.concat(updated_queries, dim=-1)

        updated_query = nn.functional.relu(self.fc(updated_queries_ten))
        return updated_query


class ActionTransformer(nn.Module):
    def __init__(self, num_units: int, embed_dim: int, key_embed_dim: int, value_embed_dim: int):
        super(ActionTransformer, self).__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.key_embed_dim = key_embed_dim
        self.value_embed_dim = value_embed_dim
        self.tx_layers: List[TxLayer] = [TxLayer(num_units=num_units, embed_dim=embed_dim, key_embed_dim=key_embed_dim,
                                                 value_embed_dim=value_embed_dim) for _ in range(0, num_units, 1)]

    def forward(self, query: Tensor, keys: List[Tensor], values: List[Tensor]):
        for tx_layer in self.tx_layers:
            query = tx_layer.forward(query, keys, values)

        return query


class OCEANRegressor(nn.Module):
    def __init__(self, embed_dim: int, tx_layer_units: int, other_interlocutor: bool = False):
        super(OCEANRegressor, self).__init__()
        self.audio_feature_dim = 100
        self.embed_dim = embed_dim
        self.num_classes = 5
        self.head: nn.Module = nn.Linear(in_features=embed_dim, out_features=self.num_classes)
        self.audio_feat_extractor: nn.Module = AudioProcessingNetwork(requires_grad=True, sample_rate=44100)
        self.query_preproc: nn.Module = QueryPreprocessor(embed_dim=embed_dim)
        self.tx_query_preproc: nn.Module = TxQueryPreprocessor(in_features=148,embed_dim=128)
        self.tx_key_value_preprocessor: nn.Module = TxKeyValPreprocessor(embed_dim, k_input_dim=embed_dim + 20 + self.audio_feature_dim ,
                                                                         v_input_dim=embed_dim + 20 + self.audio_feature_dim )
        self.video_feat_extractor: nn.Module = VideoResnetFeatureExtractor()
        self.face_features_extractor: nn.Module = VideoResnetFeatureExtractor()
        self.action_transformer: ActionTransformer = ActionTransformer(num_units=tx_layer_units, embed_dim=embed_dim,
                                                                       key_embed_dim=embed_dim,
                                                                       value_embed_dim=embed_dim)
    def create_dummy_input(self,example_wav_path: Path) -> Dict[str,Union[Tensor,np.ndarray]]:
        wf = read_wav(str(example_wav_path))
        wavenp = np.array(wf[1], dtype=float)
        sample_rate = wf[0]

        wavenp = [wavenp[:132000]]

        input_dict: Dict[str, Tensor] = {
            "local_metadata": torch.ones((1, 20)),
            "audio_chunks": wavenp,
            "face_chunks": torch.ones((1, 32, 3, 112, 112)),
            "local_context_chunks": torch.ones((1, 32, 3, 112, 112))
        }
        return input_dict

    def forward(self, input_tensors: Dict[str, Tensor]):

        batch_size = input_tensors["face_chunks"].shape[0]
        l_mtdt = input_tensors["local_metadata"]
        l_mtdt = torch.stack(batch_size * [l_mtdt])

        # audio_processing
        audio_features = self.audio_feat_extractor(input_tensors["audio_chunks"])
        #

        # Query processing
        face_features = self.face_features_extractor(input_tensors["face_chunks"])
        preprocessed_query = self.query_preproc(face_features)
        preprocessed_query = preprocessed_query
        preprocessed_query = self.tx_query_preproc.forward(preprocessed_query,l_mtdt)
        #

        # Video chunks processing
        ctx_features = self.video_feat_extractor(input_tensors["local_context_chunks"])
        audio_features = audio_features.repeat(1, 16, 28, 28, 1)
        mm_features = torch.cat([ctx_features, audio_features], dim=-1)

        lc_key, lc_value = self.tx_key_value_preprocessor(mm_features)
        transformer_out = self.action_transformer(preprocessed_query, lc_key, lc_value)
        ocean_scores = self.head(transformer_out)

        return ocean_scores



if __name__ == "__main__":

    """model = VideoResnetFeatureExtractor()
    a = torch.normal(0,1,[1,3,32,112,112])
    out = model.forward(a)
    #ts.summary(model, input_size=(3, 32, 112, 112))
    """

    """
    out = model(a)
    print(out.shape)

    model2 = QueryPreprocessor()
    a = t.ones([2,148,16,28,28])
    out = model2(a)

    
    a=STENetwork()
    print(a().shape)
    """

    """
    import numpy as np
    from scipy.io.wavfile import read as read_wav

    wf = read_wav("bus_chatter.wav")
    wavenp = np.array(wf[1],dtype=float)
    sample_rate = wf[0]

    wavenp = wavenp[:132000]
    a = AudioProcessingNetwork(requires_grad=False,sample_rate=sample_rate)
    a.forward(wavenp)


    txns = [TxLayer(num_units=2,embed_dim=128,key_embed_dim=128,value_embed_dim=128) for _ in range(2)]

    lk1 = torch.normal(0,1,[1,16,28,28,128])
    lk2 = torch.normal(0,1,[1,16,28,28,128])
    lv1 = torch.normal(0,1,[1,16,28,28,128])
    lv2 = torch.normal(0,1,[1,16,28,28,128])
    lks = [lk1,lk2]
    lvs = [lv1,lv2]
    updated_query = torch.normal(0,1,[1,128])

    for tx in txns:
        updated_query = tx.forward(updated_query,lks,lvs)
    """

    wf = read_wav("bus_chatter.wav")
    wavenp = np.array(wf[1], dtype=float)
    sample_rate = wf[0]

    wavenp = [wavenp[:132000]]

    ocean_regressor = OCEANRegressor(embed_dim=128, tx_layer_units=1)

    input_dict: Dict[str, Tensor] = {
        "local_metadata": torch.ones((1, 20)),
        "audio_chunks": wavenp,
        "face_chunks": torch.ones((1, 32, 3, 112, 112)),
        "local_context_chunks": torch.ones((1, 32, 3,112, 112))
    }
    
    out = ocean_regressor(input_dict)