"""Build standalone training YAMLs from the paper's settings and experiment grids."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import yaml

from any2graph_v2.parameter import parse_training_experiment_parameters

ROOT = Path(__file__).resolve().parents[1]
CSV_ROOT = ROOT / 'experiments/paper/csv'
OUTPUT = ROOT / 'experiments/paper/configs'
LOSS_NAMES = {
    'J(T;prediction,target)': 'original',
    'Ja(T;prediction,target)': 'alt_a',
    'Jb(T;prediction,target)': 'alt_b',
    'Ja(T;target,prediction)': 'alt_a_prime',
    'Jb(T;target,prediction)': 'alt_b_prime',
}


def rows(name):
    with (CSV_ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def build():
    settings = {r['parameter']: r for r in rows('table4_hyperparameters.csv')}
    for column in ('molecular_tasks', 'coloring'):
        for key, expected in {'optimizer': 'AdamW', 'precision': 'FP32',
                              'schedule': '5% warmup + cosine',
                              'hardware': '1 NVIDIA V100'}.items():
            if settings[key][column] != expected:
                raise ValueError(f'Unsupported Table 4 setting: {key}={settings[key][column]!r}')
    budgets = {int(r['max_nodes']): r for r in rows('table5_coloring_budgets.csv')}
    sizes = {r['dataset']: r for r in rows('table3_dataset_statistics.csv')}
    bases = {name: asdict(parse_training_experiment_parameters(['--config', name]))
             for name in ('default', 'coloring')}
    output = {}
    entries = []

    def base(task):
        coloring = task.startswith('Coloring')
        column = 'coloring' if coloring else 'molecular_tasks'
        c = copy.deepcopy(bases['coloring' if coloring else 'default'])
        n = int(sizes[task]['max_nodes'])
        c['data'].update(n_nodes_max=n, seed=0, task='coloring2graph' if coloring else
                         ('ms2graph' if task == 'MS2Scaffold' else 'fingerprint2graph'))
        c['data']['scaffold'] = task == 'MS2Scaffold'
        if coloring:
            c['data'].update(data_dir=f'src/data/coloring/image_max{n}', data_file=f'src/data/coloring/image_max{n}/manifest.json',
                             batch_size=int(budgets[n]['batch_size']))
        else:
            c['data'].update(batch_size=int(settings['batch_size'][column]),
                             data_dir='src/data/massspecgym',
                             data_file='src/data/fingerprint2graph/PUBCHEM_32.csv',
                             remove_invalid_molecules=True)
        c['model'].update(
            d_token_input=int(settings['encoder_dimension'][column]),
            d_node_decoder=int(settings['node_decoder_width'][column]),
            decoder_layers=int(settings['node_decoder_layers'][column]),
            encoder_layers=3,
            n_heads=int(settings['attention_heads'][column]),
            dropout=float(settings['dropout'][column]),
            d_token_input_feedforward=2 * int(settings['encoder_dimension'][column]),
            d_node_decoder_feedforward=2 * int(settings['node_decoder_width'][column]),
        )
        if coloring:
            c['model']['coloring_image_encoder_type'] = 'segformer_b1'
        c['target_encoder'].update(
            d_node_target=int(settings['target_gnn_width'][column]),
            target_layers=int(settings['target_gnn_layers'][column]),
            target_dropout=float(settings['dropout'][column]),
            laplacian_pe_dim=int(settings['laplacian_pe_dimension'][column]),
        )
        c['matcher'].update(matcher_dim=int(settings['matcher_dimension'][column]),
                            sinkhorn_iterations=int(settings['sinkhorn_iterations'][column]),
                            matcher_epsilon=3e-5 if not coloring else 3e-5 * (10/n)**2)
        c['solver']['old_solver_approach'] = False
        c['objective'].update(reconstruction_loss='original', alpha_marginal=1.0,
                               feature_diffusion=not coloring)
        epoch_text = settings['max_epochs']['molecular_tasks']
        epochs = int(budgets[n]['epochs']) if coloring else int(
            next(part.strip().split()[0] for part in epoch_text.split('/') if task in part))
        c['training'].update(max_epochs=epochs, early_stopping_patience=epochs,
            learning_rate=float(settings['learning_rate'][column]),
            min_learning_rate=float(settings['minimum_learning_rate'][column]),
            lr_scheduler='warmup_cosine', lr_warmup_fraction=0.05,
            gradient_clip_val=float(settings['gradient_clip'][column]),
            precision='32-true', accelerator='gpu', devices=1,
            enable_timing_metrics=True, timing_warmup_steps=20,
            wandb_mode='disabled')
        return c

    def mirror(c, inner=20, outer=20, tau=0.1):
        c['solver'].update(old_solver_approach=True, solver_type='mirror',
                           solver_backend='gpu', max_iter_inner=inner,
                           max_iter_outer=outer, tau=tau)

    def emit(name, c, source, row, **metadata):
        c['training'].update(run_name=name, output_dir=f'artifacts/paper/{name}')
        # Paths become portable YAML scalars; nulls retain parser defaults.
        document = json.loads(json.dumps(c, default=str))
        output[name + '.yaml'] = '# Generated by scripts/build_paper_configs.py; see ../README.md.\n' + yaml.safe_dump(document, sort_keys=False)
        entries.append(dict(id=name, config=f'experiments/paper/configs/{name}.yaml',
                            source_csv=source, source_row=row, **metadata))

    for i, r in enumerate(rows('table1_main_results.csv'), 1):
        if r['model'] == 'FGWBARY':
            continue
        c = base(r['task'])
        method = {'Any2Graph': 'mirror', 'Any2Graph + Matcher': 'matcher',
                  'Relationformer': 'relationformer'}[r['model']]
        if method == 'mirror':
            mirror(c)
            c['training']['max_epochs'] //= 2
            c['training']['early_stopping_patience'] = c['training']['max_epochs']
        elif method == 'relationformer':
            c['model']['graph_decoder_type'] = 'relationformer'
        emit(f'main-{r["task"].lower()}-{method}', c, 'table1_main_results.csv', i,
             implementation_note='Relationformer decoder with this repository\'s learned matching/objective' if method == 'relationformer' else '')

    for i, r in enumerate(rows('table2_loss_ablation.csv'), 1):
        c = base(r['task'])
        loss = LOSS_NAMES[r['paper_loss']]
        c['objective']['reconstruction_loss'] = loss
        emit(f'loss-{r["task"].lower()}-{loss}', c, 'table2_loss_ablation.csv', i,
             paper_loss=r['paper_loss'], implementation_loss=loss)

    for i, r in enumerate(rows('table6_frontier_grid.csv'), 1):
        c = base('Coloring20')
        if r['method'] == 'Mirror':
            mirror(c, int(r['inner_iterations']), int(r['outer_iterations']), float(r['tau']))
            name = f'frontier-mirror-i{r["inner_iterations"]}-o{r["outer_iterations"]}'
        else:
            c['matcher'].update(matcher_epsilon=float(r['epsilon']), sinkhorn_iterations=int(r['inner_iterations']))
            c['objective']['alpha_marginal'] = float(r['marginal_kl'] == 'true')
            name = f'frontier-matcher-i{r["inner_iterations"]}-kl{int(c["objective"]["alpha_marginal"])}'
        emit(name, c, 'table6_frontier_grid.csv', i, included_in_plot=r['included_in_plot'] == 'true')

    for i, r in enumerate(rows('table7_epsilon_candidates.csv'), 1):
        c = base('Coloring' + r['max_nodes'])
        c['matcher']['matcher_epsilon'] = float(Decimal(r['epsilon_in_units_1e_minus6']) * Decimal('0.000001'))
        emit(f'scaling-n{r["max_nodes"]}-matcher-x{r["center_multiplier"]}', c,
             'table7_epsilon_candidates.csv', i)

    for i, r in enumerate(rows('figure2_scaling_results_template.csv'), 1):
        if r['method'] != 'Mirror':
            continue  # No selected matcher epsilon: use the Table 7 candidates.
        c = base('Coloring' + r['max_nodes'])
        mirror(c, int(r['inner_iterations']), int(r['outer_iterations']), float(r['tau']))
        c['training']['max_epochs'] //= 2
        c['training']['early_stopping_patience'] = c['training']['max_epochs']
        emit(f'scaling-n{r["max_nodes"]}-mirror', c, 'figure2_scaling_results_template.csv', i)
    data_commands = []
    for name, row in sizes.items():
        if name.startswith('Coloring'):
            total = int(row['samples'])
            held_out = total // 10
            data_commands.append(dict(dataset=name, total_samples=total, command=[
                'uv', 'run', '--frozen', 'python', '-m', 'any2graph_v2.generate_coloring',
                '--output-dir', f'src/data/coloring/image_max{row["max_nodes"]}',
                '--min-nodes', row['min_nodes'], '--max-nodes', row['max_nodes'],
                '--train-samples', str(total - 2 * held_out),
                '--val-samples', str(held_out), '--test-samples', str(held_out), '--seed', '0']))
    output['index.json'] = json.dumps({'experiments': entries, 'data_generation': data_commands,
        'not_generated': {'FGWBARY': 'No implementation in this repository',
                          'selected_scaling_matcher': 'Selected epsilon is absent; run the Table 7 candidates'}}, indent=2) + '\n'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check committed configs without writing files')
    args = parser.parse_args()
    expected = build()
    if args.check:
        stale = [name for name, text in expected.items() if not (OUTPUT/name).exists() or (OUTPUT/name).read_text() != text]
        extras = {p.name for p in OUTPUT.glob('*.yaml')} - expected.keys()
        if stale or extras:
            parser.error(f'Configs need regeneration: {stale + sorted(extras)}')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, text in expected.items():
            (OUTPUT/name).write_text(text)
    for name in expected:
        if name.endswith('.yaml'):
            parse_training_experiment_parameters(['--config', str(OUTPUT/name)])
    print(f'{len(expected)-1} experiment configs validated.')


if __name__ == '__main__':
    main()
