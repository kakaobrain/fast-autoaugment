from typing import Optional, Tuple, List

from overrides import EnforceOverrides

from ..common.config import Config
from .model_desc import ModelDesc, OpDesc, CellType, NodeDesc, \
                        EdgeDesc, CellDesc, AuxTowerDesc, RunMode

class ModelDescBuilder(EnforceOverrides):
    def __init__(self, conf_data: Config, conf_model_desc: Config,
                 run_mode:RunMode, template:Optional[ModelDesc]=None)->None:
        self.ds_name = conf_data['name']
        self.ch_in = conf_data['ch_in']
        self.n_classes = conf_data['n_classes']

        self.init_ch_out = conf_model_desc['init_ch_out']
        self.n_cells = conf_model_desc['n_cells']
        self.n_nodes = conf_model_desc['n_nodes']
        self.n_out_nodes = conf_model_desc['n_out_nodes']
        self.stem_multiplier = conf_model_desc['stem_multiplier']
        self.aux_weight = conf_model_desc['aux_weight']
        self.drop_path_prob = conf_model_desc['drop_path_prob']
        self.run_mode = run_mode
        self.template = template

        self._set_templates()
        self._set_op_names()

    def _set_templates(self)->None:
        self.normal_template,  self.reduction_template = None, None
        if self.template is not None:
            for cell_desc in self.template.cell_descs:
                if self.normal_template is None and \
                        cell_desc.cell_type==CellType.Regular:
                  self.normal_template = cell_desc
                if self.reduction_template is None and \
                        cell_desc.cell_type==CellType.Reduction:
                    self.reduction_template = cell_desc

    def get_model_desc(self)->ModelDesc:
        stem0_op, stem1_op = self._get_stem_ops()
        assert stem0_op.ch_out is not None
        cell_descs = self._get_cell_descs(stem0_op.ch_out)

        ch_out = cell_descs[-1].get_ch_out()
        pool_op = OpDesc(self.pool_op_name, self.run_mode,
                         ch_in=ch_out, ch_out=ch_out)

        return ModelDesc(stem0_op, stem1_op, pool_op,
                         self.ch_in, self.n_classes, cell_descs)

    def _set_op_names(self)->None:
        if self.ds_name == 'cifar10':
            self.stem0_op_name, self.stem1_op_name, self.pool_op_name = \
                'stem_cifar', 'stem_cifar', 'pool_cifar'
        elif self.ds_name == 'imagenet':
            self.stem0_op_name, self.stem1_op_name, self.pool_op_name = \
                 'stem0_imagenet', 'stem1_imagenet', 'pool_imagenet'
        else:
            raise NotImplementedError(
                f'Stem and pool ops for "{self.ds_name}" are not supported yet')

    def _get_cell_descs(self, stem_ch_out:int)->List[CellDesc]:
        cell_descs = []
        reduction_p = False
        first_normal, first_reduction = -1, -1
        pp_ch_out, p_ch_out, ch_out = stem_ch_out, stem_ch_out, self.init_ch_out

        for ci in range(self.n_cells):
            reduction = self._is_reduction(ci)

            ch_out, first_normal, first_reduction, \
            cell_type, alphas_from =               \
                self._update_cell_vars(ci, reduction, ch_out,
                        first_normal, first_reduction)

            s0_op, s1_op = self._get_s_ops(
                ch_out, p_ch_out, pp_ch_out, reduction_p)

            nodes:List[NodeDesc] = [NodeDesc(edges=[]) for _ in range(self.n_nodes)]
            aux_tower_desc = self._get_aux_tower_desc(ci, self.n_out_nodes*ch_out)

            cell_descs.append(CellDesc(
                cell_type=cell_type, index=ci,
                nodes=nodes,
                s0_op=s0_op, s1_op=s1_op, aux_tower_desc=aux_tower_desc,
                n_out_nodes=self.n_out_nodes,
                n_node_channels=ch_out,
                alphas_from=alphas_from,
                run_mode=self.run_mode
            ))

            self._add_template_nodes(cell_descs[-1])

            # we concate all channels so next cell node gets channels from all nodes
            pp_ch_out, p_ch_out = p_ch_out, cell_descs[-1].get_ch_out()
            reduction_p = reduction

        return cell_descs

    def _add_template_nodes(self, cell_desc:CellDesc)->None:
        if self.template is None:
            return

        ch_out = cell_desc.n_node_channels
        reduction = cell_desc.cell_type == CellType.Reduction
        cell_template = self.reduction_template if reduction else self.normal_template

        if cell_template is None:
            return

        for node, template_node in zip(cell_desc.nodes, cell_template.nodes):
            for template_edge in template_node.edges:
                op_desc = OpDesc(template_edge.op_desc.name,
                                    run_mode=self.run_mode,
                                    ch_in=ch_out,
                                    ch_out=ch_out,
                                    stride=template_edge.op_desc.stride,
                                    affine=cell_desc.run_mode!=RunMode.Search)
                edge = EdgeDesc(op_desc, len(node.edges),
                                input_ids=template_edge.input_ids,
                                from_node=template_edge.from_node,
                                to_state=template_edge.to_state)
                node.edges.append(edge)

    def _is_reduction(self, cell_index:int)->bool:
        return cell_index in [self.n_cells//3, 2*self.n_cells//3]

    def _update_cell_vars(self, cell_index:int, reduction:bool, ch_out:int,
                       first_normal:int, first_reduction:int)\
                           ->Tuple[int, int, int, CellType, int]:
        if reduction:
            ch_out, cell_type = ch_out*2, CellType.Reduction
        else:
            cell_type = CellType.Regular

        if cell_type == CellType.Regular:
            first_normal = cell_index if first_normal < 0 else first_normal
            alphas_from = first_normal
        elif cell_type == CellType.Reduction:
            first_reduction = cell_index if first_reduction < 0 else first_reduction
            alphas_from = first_reduction
        else:
            raise NotImplementedError(
                f'CellType {cell_type} is not implemented')

        return  ch_out, first_normal, first_reduction, cell_type, alphas_from

    def _get_s_ops(self, ch_out: int, p_ch_out: int, pp_ch_out:int,
                   reduction_p: bool)->Tuple[OpDesc, OpDesc]:
        # TODO: investigate why affine=False for search but True for test
        if reduction_p:
            s0_op = OpDesc('prepr_reduce', run_mode=self.run_mode,
                            ch_in=pp_ch_out, ch_out=ch_out, affine=False)
        else:
            s0_op = OpDesc('prepr_normal', run_mode=self.run_mode,
                            ch_in=pp_ch_out, ch_out=ch_out,
                            affine=self.run_mode!=RunMode.Search)
        s1_op = OpDesc('prepr_normal', run_mode=self.run_mode,
                        ch_in=p_ch_out, ch_out=ch_out,
                        affine=self.run_mode!=RunMode.Search)

        return s0_op, s1_op

    def _get_aux_tower_desc(self, cell_index:int, ch_in:int) -> Optional[AuxTowerDesc]:
        aux_tower_desc = None
        # TODO: shouldn't we adding aux tower at *every* 1/3rd?
        if self.run_mode==RunMode.EvalTrain                        \
                and self.aux_weight > 0.            \
                and cell_index == 2*self.n_cells//3:
            aux_tower_desc = AuxTowerDesc(ch_in, self.n_classes, self.aux_weight)
        return aux_tower_desc

    def _get_stem_ops(self)->Tuple[OpDesc, OpDesc]:
        # TODO: weired not always use two different stemps as in original code
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine
        stem_ch_out = self.init_ch_out*self.stem_multiplier
        stem0_op = OpDesc(name=self.stem0_op_name, run_mode=self.run_mode,
                          ch_in=self.ch_in, ch_out=stem_ch_out,
                          affine=self.run_mode!=RunMode.Search)
        stem1_op = OpDesc(name=self.stem1_op_name, run_mode=self.run_mode,
                          ch_in=self.ch_in, ch_out=stem_ch_out,
                          affine=self.run_mode!=RunMode.Search)

        return stem0_op, stem1_op
