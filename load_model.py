from pickle import load

import numpy as np
import tensorflow as tf
from scipy.special import softmax
from simplecrypt import decrypt
from lib.utils.mapping_utils import get_char_by_index
from lib.config.settings import GENERAL_TEXT_MAPPING_PATH, GENERAL_TEXT_MODEL_PATH


class TrainedModel():

    def __init__(self):
        pass

    def load_graph(self, model_file_path, encrypt_password=None):
        """
        Load tensorflow graph
        Arguments:
            model_file_path {[type]} -- [description]

        Keyword Arguments:
            use_encrypted_pb {bool} -- [description] (default: {False})
            encrypt_password {str} -- [description] (default: {''})

        Returns:
            [type] -- [description]
        """
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_file_path, "rb") as model_file:
            if encrypt_password is not None:
                graph_def.ParseFromString(
                    decrypt(encrypt_password, model_file.read()))
            else:
                graph_def.ParseFromString(model_file.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph


class AOCRModel(TrainedModel):

    def __init__(self, model_path='', mapping_path=''):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.graph = self.load_graph(model_path)
        self.sess = tf.Session(graph=self.graph, config=config)
        self.dictionary = load(open(mapping_path, 'rb'))

    def cnn_part(self, images):

        input_cnn_name = 'import/img_data'
        output_cnn_name = 'import/Squeeze'
        input_cnn = self.graph.get_operation_by_name(input_cnn_name).outputs[0]
        output_cnn = self.graph.get_operation_by_name(output_cnn_name).outputs[0]
        cnn_outputs = []
        for image in images:
            
            image = np.expand_dims(image, axis=0)
            print(image.shape)
            cnn_output = self.sess.run(output_cnn, feed_dict={input_cnn: np.array(image)})
            cnn_outputs.append(cnn_output)

        return cnn_outputs

    def data_gen_part(self, cnn_outputs, max_width=400):

        for i, cnn_output in enumerate(cnn_outputs):
            padding_len = max_width - cnn_output.shape[1]
            padding_zero = np.zeros(shape=(1, padding_len, 512))
            cnn_output_add = np.concatenate((cnn_output, padding_zero), axis=1)
            cnn_output_add = cnn_output_add.transpose([1, 0, 2])
            if i == 0:
                cnn_output_adds = cnn_output_add
            else:
                cnn_output_adds = np.concatenate((cnn_output_adds, cnn_output_add), axis=1)
            encoder_mask = np.concatenate(
                (np.ones([1, cnn_output.shape[1]], dtype=np.float32),
                 np.zeros([1, padding_len], dtype=np.float32)),
                axis=1)
            if i == 0:
                encoder_masks = encoder_mask
            else:
                encoder_masks = np.concatenate((encoder_masks, encoder_mask), axis=0)

        mask_inputs = [a[:, np.newaxis] for a in encoder_masks.T]

        return cnn_output_adds, mask_inputs

    def predict(self, images, max_width=400, word_len=40, batch_size=1):

        cnn_outputs = self.cnn_part(images)
        cnn_output_adds, mask_inputs = self.data_gen_part(cnn_outputs, max_width=max_width)

        print(cnn_output_adds.shape, mask_inputs.shape)

        input_feed = {}

        input_feed[self.graph.get_operation_by_name('import/transpose_1').outputs[0]] = cnn_output_adds
        input_feed[self.graph.get_operation_by_name('import/decoder0').outputs[0]] = [1] * batch_size

        for l in range(max_width):
            # print(np.array(mask_inputs).shape)
            input_feed[self.graph.get_operation_by_name('import/encoder_mask' + str(l)).outputs[0]] = mask_inputs[l]
        output_feed = []
        for l in range(word_len + 2):  # Output logits.
            if l == 0:
                output_feed.append(self.graph.get_operation_by_name(
                    'import/model_with_buckets/embedding_attention_decoder/attention_decoder/AttnOutputProjection/AttnOutputProjection/BiasAdd').outputs[
                                       0])
            else:
                output_feed.append(self.graph.get_operation_by_name(
                    'import/model_with_buckets/embedding_attention_decoder/attention_decoder/AttnOutputProjection_{}/AttnOutputProjection/BiasAdd'.format(
                        str(
                            l))).outputs[0])
        for l in range(word_len + 2):  # Output logits.
            if l == 0:
                output_feed.append(self.graph.get_operation_by_name(
                    'import/model_with_buckets/embedding_attention_decoder/attention_decoder/Attention_0/Softmax').outputs[
                                       0])
            else:
                output_feed.append(self.graph.get_operation_by_name(
                    'import/model_with_buckets/embedding_attention_decoder/attention_decoder/Attention_0_{}/Softmax'.format(
                        str(
                            l))).outputs[0])

        outputs = self.sess.run(output_feed, input_feed)
        step_logits = outputs[:word_len + 2]
        step_outputs = [b for b in
                        np.array([np.argmax(logit, axis=1).tolist() for logit in step_logits]).transpose()]

        # Get output
        step_probs = np.array(step_logits).transpose([1, 0, 2])
        output_texts = []
        output_probs = []
        for i, step_output in enumerate(step_outputs):
            step_prob = step_probs[i]
            output_text = ''
            probs = []
            for j, c in enumerate(step_output.tolist()):
                if c == 2:
                    break
                output_text += get_char_by_index(c - 3, self.dictionary)
                probs.append(np.max(softmax(step_prob[j])))
            output_texts.append(output_text)
            output_probs.append(probs)

        # Get attention mask
        step_attentions = outputs[word_len + 2:]
        step_attns = np.array([[a.tolist() for a in step_attn] for step_attn in step_attentions]).transpose(
            [1, 0, 2])
        word_locations = []
        for i in range(len(output_texts)):
            attentions = step_attns[i]
            locations = []
            for index in range(len(output_texts[i])):
                attention = attentions[index][:]
                location = int((np.argmax(attention)) * 4)
                locations.append(location)
            word_locations.append(locations)

        return output_texts, output_probs, word_locations


class GeneralModel():

    def __init__(self):
        self._general_binary_model = AOCRModel(model_path=GENERAL_TEXT_MODEL_PATH,
                                               mapping_path=GENERAL_TEXT_MAPPING_PATH)

    def general_line_predict(self, thresh_images, batch_size):
        line_thresh_texts, thresh_probs, word_locations = self._general_binary_model.predict(thresh_images,
                                                                                             batch_size=batch_size)
        return line_thresh_texts, word_locations
