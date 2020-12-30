
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use ndarray::Dim;
use numpy::{IntoPyArray, PyArray};

use rand::{SeedableRng, rngs::StdRng};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;


fn get_bit(number: usize, bit_index: usize) -> usize {
    number >> bit_index & 1
}

fn set_bit(number: usize, bit_index: usize, bit_value: bool) -> usize {
    if bit_value {
        number | 1 << bit_index
    } else {
        number & !(1 << bit_index)
    }
}

fn generate_nklanscape(
    num_bits: usize, num_components: usize, seed: u64
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);

    // make interaction list
    let mut interaction_lists: Vec<Vec<usize>> = Vec::with_capacity(num_bits);
    for bit_idx in 0..num_bits {
        let full_idx_list = (0..bit_idx)
            .chain(bit_idx + 1..num_bits)
            .collect::<Vec<usize>>();
        interaction_lists.push(
            full_idx_list
                .choose_multiple(&mut rng, num_components)
                .cloned()
                .collect(),
        );
    }
    for (bit_idx, interaction_list) in interaction_lists.iter_mut().enumerate() {
        interaction_list.push(bit_idx);
    }
    let interaction_lists = interaction_lists;

    // make component fitness functions
    let u_dist = Uniform::new(0f64, 1f64);
    let mut comp_fit_funcs = vec![vec![0f64; 1 << (num_components + 1)]; num_bits];

    for comp_fit_func in comp_fit_funcs.iter_mut() {
        let iterator = comp_fit_func.iter_mut().zip(u_dist.sample_iter(&mut rng));

        for (comp_config, rand_val) in iterator {
            *comp_config = rand_val;
        }
    }

    // make fitness function
    let mut fitness_func = vec![0f64; 1 << num_bits];

    for solution in 0..(1 << num_bits) {
        let mut sum_comp_funcs = 0f64;

        for bit_idx in 0..num_bits {
            let mut comp_config = 0;
            let iterator = interaction_lists[bit_idx].iter().enumerate();
            for (comp_bit_idx, dep_bit_idx) in iterator {
                comp_config = set_bit(
                    comp_config,
                    comp_bit_idx,
                    get_bit(solution, *dep_bit_idx) != 0,
                );
            }
            sum_comp_funcs += comp_fit_funcs[bit_idx][comp_config];
        }
        fitness_func[solution] = sum_comp_funcs / num_bits as f64;
    }

    let largest_fitness = fitness_func.iter().cloned().fold(0. / 0., f64::max);

    let pass_through_monotonic = false;
    if pass_through_monotonic {
        fitness_func
            .iter_mut()
            .map(|fitness| *fitness = (*fitness / largest_fitness).powf(8.))
            .last();
    } else {
        fitness_func
            .iter_mut()
            .map(|fitness| *fitness = *fitness / largest_fitness)
            .last();
    }
    fitness_func
}

fn monotonic_fn_pow8(fitness_func: &Vec<f64>) -> Vec<f64> {
    fitness_func
        .iter()
        .map(|fitness| fitness.powf(8.))
        .collect()
}


#[pymodule]
fn rustylandscapes(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "generate_fitness_func")]
    fn generate_fitness_func_py<'py>(
        py: Python<'py>, num_bits: usize, num_components: usize, seed: u64
    ) -> (&PyArray<f64, Dim<[usize; 1]>>, &PyArray<f64, Dim<[usize; 1]>>) {
        let fitness_func = generate_nklanscape(
            num_bits, num_components, seed,
        );
        let fitness_func_mono = monotonic_fn_pow8(&fitness_func);
        return (
            fitness_func_mono.into_pyarray(py),
            fitness_func.into_pyarray(py),
        )
    }

    Ok(())
}
