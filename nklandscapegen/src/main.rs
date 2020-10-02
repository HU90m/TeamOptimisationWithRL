use rand;
use rand::{SeedableRng, rngs::StdRng};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

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

fn main() -> Result<(), Box<dyn Error>> {
    let arg_err: &str = "3 arguments needed: N, K, output filename";

    let num_bits = env::args().nth(1).expect(arg_err).parse::<usize>()?; // N
    let num_components = env::args().nth(2).expect(arg_err).parse::<usize>()?; // K
    let seed = env::args().nth(3).expect(arg_err).parse::<u64>()?; // K
    let output_filename = env::args().nth(4).expect(arg_err);

    let mut rng = StdRng::seed_from_u64(seed);

    // make interaction list
    let mut interaction_lists: Vec<Vec<usize>> = Vec::new();
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

    {
        let mut buffer = File::create(output_filename)?;
        for fitness in fitness_func.iter() {
            let bytes = fitness.to_bits().to_le_bytes();
            buffer.write(&bytes)?;
        }
    }
    Ok(())
}
