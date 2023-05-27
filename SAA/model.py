import cv2
import numpy as np
import torch
from PIL import Image

# Grounding DINO, slightly modified from original repo
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# segment anything
from SAM.segment_anything import build_sam, SamPredictor
# ImageNet pretrained feature extractor
from .modelinet import ModelINet


class Model(torch.nn.Module):
    def __init__(self,
                 ## DINO
                 dino_config_file,
                 dino_checkpoint,

                 ## SAM
                 sam_checkpoint,

                 ## Parameters
                 box_threshold,
                 text_threshold,

                 ## Others
                 out_size=256,
                 device='cuda',

                 ):
        '''

        Args:
            dino_config_file: the config file for DINO
            dino_checkpoint: the path of checkpoint for DINO
            sam_checkpoint: the path of checkpoint for SAM
            box_threshold: the threshold for box filter
            text_threshold: the threshold for box filter
            out_size: the desired output resolution of anomaly map
            device: the running device, e.g, 'cuda:0'

        NOTE:
            1. In our published paper, the property prompt P^P is applied to R (region).
            Actually, we apply P^P to bounding box-level region R^B in this repo.
            2. We haven't added IoU constraint in this repo.
            3. This module only accepts BS=1.
        '''
        super(Model, self).__init__()

        # Build Model
        self.anomaly_region_generator = self.load_dino(dino_config_file, dino_checkpoint, device=device)
        self.anomaly_region_refiner = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.visual_saliency_extractor = ModelINet(device=device)

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        # Parameters
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Others
        self.out_size = out_size
        self.device = device
        self.is_sam_set = False

    def load_dino(self, model_config_path, model_checkpoint_path, device) -> torch.nn.Module:
        '''

        Args:
            model_config_path:
            model_checkpoint_path:
            device:

        Returns:

        '''
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        model = model.to(device)
        return model

    def get_grounding_output(self, image, caption, device="cpu") -> (torch.Tensor, torch.Tensor, str):
        caption = caption.lower()
        caption = caption.strip()

        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)

        with torch.no_grad():
            outputs = self.anomaly_region_generator(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        return boxes, logits, caption

    def set_ensemble_text_prompts(self, text_prompt_list: list, verbose=False) -> None:
        self.defect_prompt_list = [f[0] for f in text_prompt_list]
        self.filter_prompt_list = [f[1] for f in text_prompt_list]

        if verbose:
            print('used ensemble text prompts ===========')

            for d, t in zip(self.defect_prompt_list, self.filter_prompt_list):
                print(f'det prompts: {d}')
                print(f'filtered background: {t}')

            print('======================================')

    def set_property_text_prompts(self, property_prompts, verbose=False) -> None:

        self.object_prompt = property_prompts.split(' ')[7]
        self.object_number = int(property_prompts.split(' ')[5])
        self.k_mask = int(property_prompts.split(' ')[12])
        self.defect_area_threshold = float(property_prompts.split(' ')[19])
        self.object_max_area = 1. / self.object_number
        self.object_min_area = 0.
        self.similar = property_prompts.split(' ')[6]

        if verbose:
            print(f'{self.object_prompt}, '
                  f'{self.object_number}, '
                  f'{self.k_mask}, '
                  f'{self.defect_area_threshold}, '
                  f'{self.object_max_area}, '
                  f'{self.object_min_area}')

    def ensemble_text_guided_mask_proposal(self, image, object_phrase_list, filtered_phrase_list,
                                           object_max_area, object_min_area,
                                           bbox_score_thr, text_score_thr):

        size = image.shape[:2]
        H, W = size[0], size[1]

        dino_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        dino_image, _ = self.transform(dino_image, None)  # 3, h, w

        if self.is_sam_set == False:
            self.anomaly_region_refiner.set_image(image)
            self.is_sam_set = True

        ensemble_boxes = []
        ensemble_logits = []
        ensemble_phrases = []

        max_box_area = 0.

        for object_phrase, filtered_phrase in zip(object_phrase_list, filtered_phrase_list):

            ########## language prompts for region proposal
            boxes, logits, object_phrase = self.text_guided_region_proposal(dino_image, object_phrase)

            ########## property prompts for region filter
            boxes_filtered, logits_filtered, pred_phrases = self.bbox_suppression(boxes, logits, object_phrase,
                                                                                  filtered_phrase,
                                                                                  bbox_score_thr, text_score_thr,
                                                                                  object_max_area, object_min_area)
            ## in case there is no box left
            if boxes_filtered is not None:
                ensemble_boxes += [boxes_filtered]
                ensemble_logits += logits_filtered
                ensemble_phrases += pred_phrases

                boxes_area = boxes_filtered[:, 2] * boxes_filtered[:, 3]

                if boxes_area.max() > max_box_area:
                    max_box_area = boxes_area.max()

        if ensemble_boxes != []:
            ensemble_boxes = torch.cat(ensemble_boxes, dim=0)
            ensemble_logits = np.stack(ensemble_logits, axis=0)

            # denormalize the bbox
            for i in range(ensemble_boxes.size(0)):
                ensemble_boxes[i] = ensemble_boxes[i] * torch.Tensor([W, H, W, H]).to(self.device)
                ensemble_boxes[i][:2] -= ensemble_boxes[i][2:] / 2
                ensemble_boxes[i][2:] += ensemble_boxes[i][:2]

            # region 2 mask
            masks, logits = self.region_refine(ensemble_boxes, ensemble_logits, H, W)

        else:  # in case there is no box left
            masks = [np.zeros((H, W), dtype=bool)]
            logits = [0]
            max_box_area = 1

        return masks, logits, max_box_area

    def text_guided_region_proposal(self, dino_image, object_phrase):
        # directly use the output of Grounding DINO
        boxes, logits, caption = self.get_grounding_output(
            dino_image, object_phrase, device=self.device
        )

        return boxes, logits, caption

    def bbox_suppression(self, boxes, logits, object_phrase, filtered_phrase,
                         bbox_score_thr, text_score_thr,
                         object_max_area, object_min_area,
                         with_logits=True):

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]

        # filter the bounding boxes according to the box similarity and the area

        # strategy1: bbox score thr
        box_score_mask = logits_filt.max(dim=1)[0] > bbox_score_thr

        # strategy2: max area
        box_max_area_mask = boxes_area < (object_max_area)

        # strategy3: min area
        box_min_area_mask = boxes_area > (object_min_area)

        filt_mask = torch.bitwise_and(box_score_mask, box_max_area_mask)
        filt_mask = torch.bitwise_and(filt_mask, box_min_area_mask)

        if torch.sum(filt_mask) == 0:  # in case there are no matches
            return None, None, None
        else:
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = self.anomaly_region_generator.tokenizer
        tokenized = tokenlizer(object_phrase)

        # build pred
        pred_phrases = []
        boxes_filtered = []
        logits_filtered = []
        for logit, box in zip(logits_filt, boxes_filt):
            # strategy4: text score thr
            pred_phrase = get_phrases_from_posmap(logit > text_score_thr, tokenized, tokenlizer)

            # strategy5: filter background
            if pred_phrase.count(filtered_phrase) > 0:  # we don't want to predict the category
                continue

            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

            boxes_filtered.append(box)
            logits_filtered.append(logit.max().item())

        if boxes_filtered == []:
            return None, None, None

        boxes_filtered = torch.stack(boxes_filtered, dim=0)

        return boxes_filtered, logits_filtered, pred_phrases

    def region_refine(self, boxes_filtered, logits_filtered, H, W):
        if boxes_filtered == []:
            return [np.zeros((H, W), dtype=bool)], [0]

        transformed_boxes = self.anomaly_region_refiner.transform.apply_boxes_torch(boxes_filtered, (H, W)).to(
            self.device)

        masks, _, _ = self.anomaly_region_refiner.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks.cpu().squeeze(1).numpy()

        return masks, logits_filtered

    def saliency_prompting(self, image, object_masks, defect_masks, defect_logits):

        ###### Self Similarity Calculation
        similarity_map = self.visual_saliency_calculation(image, object_masks)

        ###### Rescore
        defect_masks, defect_rescores = self.rescore(defect_masks, defect_logits, similarity_map)

        return defect_masks, defect_rescores, similarity_map

    def single_object_similarity(self, image, object_masks):
        # use GPU version...
        # only consider the feautures of objects

        # as calculate whole image similarity is memory costly, we use a small resolution here...
        self.visual_saliency_extractor.set_img_size(256)
        resize_image = cv2.resize(image, (256, 256))
        features, ratio_h, ratio_w = self.visual_saliency_extractor(resize_image)

        B, C, H, W = features.shape
        assert B == 1
        features_flattern = features.view(B * C, H * W)

        features_self_similarity = features_flattern.T @ features_flattern
        features_self_similarity = 0.5 * (1 - features_self_similarity)

        features_self_similarity = features_self_similarity.sort(dim=1, descending=True)[0]

        # by default we use N=400 for saliency calculation
        features_self_similarity = torch.mean(features_self_similarity[:, :400], dim=1)
        heatMap2 = features_self_similarity.view(H, W).cpu().numpy()

        mask_anomaly_scores = cv2.resize(heatMap2, (image.shape[1], image.shape[0]))
        # mask_anomaly_scores[~object_masks] = 0.
        return mask_anomaly_scores

    def visual_saliency_calculation(self, image, object_masks):

        if self.object_number == 1:  # use single-instance strategy
            mask_area = np.sum(object_masks, axis=(1, 2))
            object_mask = object_masks[mask_area.argmax(), :, :]
            self_similarity_anomaly_map = self.single_object_similarity(image, object_mask)
            return self_similarity_anomaly_map

        else:  # use multi-instance strategy
            resize_image = cv2.resize(image, (1024, 1024))
            features, ratio_h, ratio_w = self.visual_saliency_extractor(resize_image)

            feature_size = features.shape[2:]
            object_masks_clone = object_masks.copy()
            object_masks_clone = object_masks_clone.astype(np.int32)

            resize_object_masks = []
            for object_mask in object_masks_clone:
                resize_object_masks.append(cv2.resize(object_mask, feature_size, interpolation=cv2.INTER_NEAREST))

            mask_anomaly_scores = []

            for indx in range(len(resize_object_masks)):
                other_object_masks1 = resize_object_masks[:indx]
                other_object_masks2 = resize_object_masks[indx + 1:]
                other_object_masks = other_object_masks1 + other_object_masks2

                one_mask_feature, \
                one_feature_location, \
                other_mask_features = self.region_feature_extraction(
                    features,
                    resize_object_masks[indx],
                    other_object_masks
                )

                similarity = one_mask_feature @ other_mask_features.T  # (H*W, N)
                similarity = similarity.max(dim=1)[0]
                anomaly_score = 0.5 * (1. - similarity)
                anomaly_score = anomaly_score.cpu().numpy()

                mask_anomaly_score = np.zeros(feature_size)
                for location, score in zip(one_feature_location, anomaly_score):
                    mask_anomaly_score[location[0], location[1]] = score

                mask_anomaly_scores.append(mask_anomaly_score)

            mask_anomaly_scores = np.stack(mask_anomaly_scores, axis=0)
            mask_anomaly_scores = np.max(mask_anomaly_scores, axis=0)
            mask_anomaly_scores = cv2.resize(mask_anomaly_scores, (image.shape[1], image.shape[0]))

            return mask_anomaly_scores

    def region_feature_extraction(self, features, one_object_mask, other_object_masks):
        '''
        Use ImageNet pretraine network to extract features for mask
        Args:
            features:
            one_object_mask:
            other_object_masks:

        Returns:

        '''
        features_clone = features.clone()
        one_mask_feature = []
        one_feature_location = []
        for h in range(one_object_mask.shape[0]):
            for w in range(one_object_mask.shape[1]):
                if one_object_mask[h, w] > 0:
                    one_mask_feature += [features_clone[:, :, h, w].clone()]
                    one_feature_location += [np.array((h, w))]
                    features_clone[:, :, h, w] = 0.

        one_feature_location = np.stack(one_feature_location, axis=0)
        one_mask_feature = torch.cat(one_mask_feature, dim=0)

        B, C, H, W = features_clone.shape
        assert B == 1
        features_clone_flattern = features_clone.view(C, -1)

        other_mask_features = []
        for other_object_mask in other_object_masks:
            other_object_mask_flattern = other_object_mask.reshape(-1)
            other_mask_feature = features_clone_flattern[:, other_object_mask_flattern > 0]
            other_mask_features.append(other_mask_feature)

        other_mask_features = torch.cat(other_mask_features, dim=1).T

        return one_mask_feature, one_feature_location, other_mask_features

    def rescore(self, defect_masks, defect_logits, similarity_map):
        defect_rescores = []
        for mask, logit in zip(defect_masks, defect_logits):
            if similarity_map[mask].size == 0:
                similarity_score = 1.
            else:
                similarity_score = np.exp(3 * similarity_map[mask].mean())

            refined_score = logit * similarity_score
            defect_rescores.append(refined_score)

        defect_rescores = np.stack(defect_rescores, axis=0)

        return defect_masks, defect_rescores

    def confidence_prompting(self, defect_masks, defect_scores, similarity_map):
        mask_indx = defect_scores.argsort()[-self.k_mask:]

        filtered_masks = []
        filtered_scores = []

        for indx in mask_indx:
            filtered_masks.append(defect_masks[indx])
            filtered_scores.append(defect_scores[indx])

        anomaly_map = np.zeros(defect_masks[0].shape)
        weight_map = np.ones(defect_masks[0].shape)

        for mask, logits in zip(filtered_masks, filtered_scores):
            anomaly_map += mask * logits
            weight_map += mask * 1.

        anomaly_map[weight_map > 0] /= weight_map[weight_map > 0]
        anomaly_map = cv2.resize(anomaly_map, (self.out_size, self.out_size))
        return anomaly_map

    def forward(self, image: np.ndarray):
        ####### Object TGMP for object detection
        object_masks, object_logits, object_area = self.ensemble_text_guided_mask_proposal(
            image,
            [self.object_prompt],
            ['PlaceHolder'],
            self.object_max_area,
            self.object_min_area,
            self.box_threshold,
            self.text_threshold
        )

        ###### Reasoning: set the anomaly area threshold according to object area
        self.defect_max_area = object_area * self.defect_area_threshold
        self.defect_min_area = 0.

        ####### language prompts and property prompts $\mathcal{P}^L$ $\mathcal{P}^S$
        ####### for region proposal and filter
        defect_masks, defect_logits, _ = self.ensemble_text_guided_mask_proposal(
            image,
            self.defect_prompt_list,
            self.filter_prompt_list,
            self.defect_max_area,
            self.defect_min_area,
            self.box_threshold,
            self.text_threshold
        )

        ###### saliency prompts $\mathcal{P}^S$
        defect_masks, defect_rescores, similarity_map = self.saliency_prompting(
            image,
            object_masks,
            defect_masks,
            defect_logits
        )

        ##### confidence prompts $\mathcal{P}^C$
        anomaly_map = self.confidence_prompting(defect_masks, defect_rescores, similarity_map)

        self.is_sam_set = False

        appendix = {'similarity_map': similarity_map}

        return anomaly_map, appendix
