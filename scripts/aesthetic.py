
from modules import scripts, script_callbacks, shared
import aesthetic_clip
import gradio as gr

aesthetic = aesthetic_clip.AestheticCLIP()
aesthetic_imgs_components = []


class AestheticScript(scripts.Script):
    def title(self):
        return "Aesthetic embeddings"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative = aesthetic_clip.create_ui()

        self.infotext_fields = [
            (aesthetic_lr, "Aesthetic LR"),
            (aesthetic_weight, "Aesthetic weight"),
            (aesthetic_steps, "Aesthetic steps"),
            (aesthetic_imgs, "Aesthetic embedding"),
            (aesthetic_slerp, "Aesthetic slerp"),
            (aesthetic_imgs_text, "Aesthetic text"),
            (aesthetic_text_negative, "Aesthetic text negative"),
            (aesthetic_slerp_angle, "Aesthetic slerp angle"),
        ]

        aesthetic_imgs_components.append(aesthetic_imgs)

        return [aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative]

    def process(self, p, aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative):
        aesthetic.set_aesthetic_params(p, float(aesthetic_lr), float(aesthetic_weight), int(aesthetic_steps), aesthetic_imgs, aesthetic_slerp, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative)


def on_model_loaded(sd_model):
    aesthetic.process_tokens = sd_model.cond_stage_model.process_tokens
    sd_model.cond_stage_model.process_tokens = aesthetic


def on_script_unloaded():
    cond_stage_model = shared.sd_model.cond_stage_model
    if type(cond_stage_model.process_tokens) == aesthetic_clip.AestheticCLIP:
        cond_stage_model.process_tokens = cond_stage_model.process_tokens.process_tokens


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as aws_sd_inf:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="train your model with built-in models or create your own embedding")

                newEmbedding = gr.Textbox(label="Name")
                # dropdown with placeholder choices
                dropDown = gr.Dropdown(label="Embedding", choices=["Choose embedding"])
                textBox = gr.Textbox(label='Hyperparameters')

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(value="")

                    with gr.Column():
                        createEmbedding = gr.Button(value="Submit your training job", variant='primary')

            with gr.Column():
                output = gr.Text(value="", show_label=False)

        dropdown_components = aesthetic_imgs_components.copy()

        def generate_embs(*args):
            res = aesthetic_clip.generate_imgs_embd(*args)

            aesthetic_clip.update_aesthetic_embeddings()
            updates = [gr.Dropdown.update(choices=sorted(aesthetic_clip.aesthetic_embeddings.keys())) for _ in range(len(dropdown_components))]

            return [*updates, res]

        createEmbedding.click(
            fn=generate_embs,
            inputs=[
                newEmbedding,
                dropDown,
                textBox
            ],
            outputs=[
                *dropdown_components,
                output
            ]
        )

        aesthetic_imgs_components.clear()

    return [(aws_sd_inf, "Create AWS embedding/training job", "aws_sd_inf")]


script_callbacks.on_script_unloaded(on_script_unloaded)
script_callbacks.on_model_loaded(on_model_loaded)
script_callbacks.on_ui_tabs(on_ui_tabs)
