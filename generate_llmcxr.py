from training.generate import generate_response, load_model_tokenizer_for_generate
from training.mimiccxr_vq_dataset import sample_cxr_vq_output_instruction, sample_cxr_vq_input_instruction, CXR_VQ_TOKENIZER_LEN

from transformers import PreTrainedTokenizer

from typing import List
from argparse import ArgumentParser


def shift_vq_tokens(tokens: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    return [token + len(tokenizer) for token in tokens]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)

    instruction_texts = [
        # Nautral language instructions
        "Tell me about Independence Day in the United States.",
        "Does MRI pose a risk of radiation exposure? Please write as long as you can.",

        # CXR-to-Report instructions
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),

        # Report-to-CXR instructions
        sample_cxr_vq_output_instruction(),
        sample_cxr_vq_output_instruction(),
        sample_cxr_vq_output_instruction()
    ]
    input_texts = [
        None,
        None,

        # dicom_id: edf1e5ad-e7249deb-2d881608-aa2878c8-e22288bd
        [955, 245, 63, 127, 981, 1002, 829, 147, 665, 716, 447, 973, 533, 329, 659, 151, 61, 127, 410, 920, 439, 203, 600, 921, 202, 573, 742, 885, 687, 71, 22, 807, 22, 621, 764, 764, 66, 742, 742, 716, 807, 551, 860, 410, 15, 144, 719, 1012, 401, 764, 87, 122, 339, 716, 258, 144, 193, 551, 203, 122, 333, 764, 428, 921, 634, 921, 860, 245, 961, 637, 973, 345, 665, 63, 909, 1012, 200, 905, 322, 680, 133, 660, 322, 339, 551, 699, 22, 621, 87, 742, 250, 961, 127, 845, 333, 1012, 77, 295, 205, 203, 82, 71, 144, 468, 202, 807, 1012, 333, 69, 331, 144, 680, 468, 921, 331, 981, 955, 232, 1002, 467, 446, 551, 410, 716, 533, 845, 406, 147, 260, 514, 331, 533, 243, 955, 468, 634, 534, 742, 127, 1012, 921, 119, 600, 981, 22, 534, 528, 127, 807, 973, 634, 401, 77, 345, 404, 932, 528, 814, 243, 71, 250, 808, 885, 716, 932, 447, 634, 790, 999, 345, 428, 243, 406, 814, 514, 637, 659, 845, 1012, 468, 790, 421, 790, 468, 243, 34, 719, 824, 193, 163, 842, 250, 77, 193, 930, 295, 105, 119, 790, 119, 439, 428, 829, 147, 660, 401, 529, 699, 200, 808, 30, 133, 30, 529, 151, 200, 845, 202, 845, 660, 66, 331, 82, 961, 818, 467, 61, 785, 108, 955, 818, 163, 687, 329, 533, 528, 330, 15, 329, 961, 71, 1012, 764, 439, 304, 1012, 860, 596, 163, 842, 999, 845, 905, 514, 716, 122], 
        # dicom_id: 494f62af-2213616c-20174f23-c3d781fd-fed10e18
        [66, 260, 379, 555, 304, 63, 222, 127, 716, 258, 932, 1012, 34, 533, 885, 1012, 163, 163, 119, 63, 82, 860, 447, 383, 260, 383, 404, 1015, 15, 860, 260, 87, 981, 447, 339, 322, 105, 1015, 764, 764, 15, 905, 807, 232, 1015, 528, 1012, 127, 973, 764, 981, 600, 87, 15, 202, 829, 905, 845, 829, 932, 551, 410, 447, 999, 447, 203, 1012, 1002, 829, 1015, 829, 660, 860, 119, 764, 447, 468, 860, 322, 71, 1015, 370, 955, 621, 1002, 34, 202, 978, 22, 790, 119, 428, 534, 447, 401, 127, 63, 930, 932, 34, 829, 1002, 193, 660, 824, 529, 359, 534, 716, 829, 304, 63, 379, 551, 932, 764, 260, 129, 428, 921, 163, 200, 921, 785, 404, 63, 706, 555, 551, 961, 932, 764, 973, 406, 1012, 555, 200, 61, 77, 845, 860, 107, 295, 1012, 379, 203, 243, 764, 773, 687, 909, 446, 66, 163, 773, 406, 105, 333, 69, 716, 294, 66, 243, 764, 814, 331, 295, 304, 30, 193, 533, 785, 814, 292, 69, 1012, 151, 573, 719, 829, 528, 785, 814, 107, 292, 292, 193, 467, 193, 687, 83, 1012, 250, 999, 193, 22, 295, 905, 331, 203, 77, 77, 83, 193, 330, 222, 860, 63, 961, 961, 379, 600, 845, 345, 108, 200, 133, 66, 30, 108, 66, 660, 621, 339, 250, 699, 258, 534, 660, 845, 534, 551, 232, 77, 401, 359, 119, 34, 15, 329, 329, 742, 773, 845, 842, 634, 193, 421, 133, 22, 829, 379, 119, 331, 34, 339], 
        # dicom_id: c710e145-280390c3-5b9ddcf7-faa611b8-b39e60c8
        [250, 122, 66, 921, 680, 514, 295, 428, 921, 63, 981, 203, 122, 719, 410, 551, 108, 699, 955, 406, 514, 202, 1015, 406, 66, 63, 555, 909, 383, 243, 824, 808, 596, 410, 533, 133, 764, 232, 501, 807, 716, 447, 222, 202, 932, 329, 294, 573, 961, 200, 359, 955, 250, 555, 716, 439, 961, 203, 122, 447, 873, 873, 322, 555, 978, 665, 127, 773, 529, 873, 1012, 339, 742, 250, 250, 127, 829, 144, 333, 533, 555, 83, 699, 981, 232, 410, 34, 329, 250, 961, 329, 250, 659, 660, 680, 129, 842, 932, 203, 63, 410, 71, 665, 83, 699, 955, 961, 551, 66, 660, 665, 921, 842, 15, 773, 529, 909, 133, 295, 428, 961, 955, 981, 250, 790, 144, 514, 921, 428, 719, 243, 77, 807, 467, 842, 842, 304, 71, 250, 930, 82, 659, 842, 842, 660, 15, 600, 22, 322, 34, 845, 345, 428, 232, 909, 818, 63, 447, 814, 467, 905, 600, 467, 69, 845, 534, 785, 634, 764, 34, 105, 706, 1002, 921, 814, 808, 331, 468, 687, 660, 932, 163, 905, 790, 790, 528, 687, 534, 829, 105, 200, 808, 842, 596, 534, 528, 406, 330, 785, 790, 330, 468, 905, 785, 814, 808, 163, 66, 829, 133, 383, 292, 322, 621, 200, 133, 66, 133, 331, 790, 61, 719, 250, 930, 401, 193, 446, 621, 108, 108, 30, 61, 61, 61, 250, 205, 77, 77, 193, 22, 905, 955, 30, 330, 30, 292, 292, 292, 133, 330, 955, 258, 824, 955, 330, 634], 
        # dicom_id: 1bc3d3de-cd13c1cd-ce13e61d-5191632c-e3ae7b5c
        [719, 421, 551, 421, 742, 122, 680, 978, 921, 129, 329, 339, 200, 63, 814, 379, 151, 885, 706, 529, 428, 905, 144, 829, 147, 406, 447, 905, 533, 151, 978, 773, 200, 147, 978, 773, 232, 1012, 1012, 829, 534, 105, 716, 621, 15, 222, 406, 359, 439, 773, 127, 829, 637, 203, 379, 534, 331, 905, 742, 981, 764, 508, 203, 243, 151, 15, 508, 873, 71, 250, 370, 331, 193, 764, 501, 410, 383, 119, 596, 508, 22, 807, 147, 909, 410, 742, 468, 885, 706, 596, 329, 260, 659, 814, 468, 885, 151, 921, 322, 637, 329, 551, 119, 359, 596, 514, 339, 699, 955, 404, 127, 370, 151, 147, 1015, 63, 329, 528, 905, 404, 824, 34, 119, 122, 955, 1002, 807, 999, 955, 428, 1012, 551, 243, 932, 468, 446, 421, 107, 428, 528, 621, 660, 406, 63, 151, 69, 706, 401, 860, 932, 383, 151, 330, 34, 467, 69, 467, 905, 1012, 961, 294, 534, 428, 468, 706, 829, 66, 292, 329, 808, 193, 764, 814, 659, 82, 245, 250, 428, 808, 742, 790, 133, 108, 30, 383, 404, 439, 331, 1015, 222, 447, 501, 250, 660, 15, 205, 30, 446, 930, 331, 22, 87, 634, 764, 634, 222, 447, 370, 955, 1002, 660, 1002, 790, 232, 15, 63, 764, 1002, 467, 401, 660, 428, 339, 501, 529, 807, 428, 845, 955, 421, 706, 383, 528, 860, 200, 30, 842, 773, 203, 501, 699, 203, 665, 447, 69, 660, 621, 706, 232, 205, 845, 660, 921, 764, 329, 379],
        
        "A new dual-lead pacemaker with lead positioned through the left transvenous approach end into the right ventricle and is appropriate. No focal lung opacities concerning for pneumonia.  Heart is top normal size. Mediastinal and hilar contours are normal.  No evidence of pneumothorax."
        "Bilateral, diffuse, confluent pulmonary opacities. Differential diagnosis include  severe pulmonary edema or ARDS or hemorrhage. Concurrent lung infection cannot be ruled out.",
        "No acute cardiopulmonary process."
    ]

    generation_samples = list(zip(instruction_texts, input_texts))
    for i, (instruction_text, input_text) in enumerate(generation_samples, start=1):
        # Input image token must be shifted by tokenizer length before being passed to model.
        if isinstance(input_text, list):
            input_text = shift_vq_tokens(input_text, tokenizer)

        response, response_vq = generate_response((instruction_text, input_text), model=model, tokenizer=tokenizer, max_new_tokens=512)
        if response:
            print(f"----------- {i}/{len(generation_samples)}\n\nInstruction: {instruction_text}\n\nInput: {input_text}\n\nResponse: {response}\n\nGenerated-VQ: {response_vq}\n\n-----------\n\n\n")


if __name__ == "__main__":
    main()