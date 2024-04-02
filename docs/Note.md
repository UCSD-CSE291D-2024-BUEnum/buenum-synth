# Note

## `egraph.split_eclass`

This method should should be invoked in the `enumerator.rebuild(pts)` method, where the new `pts` would replace the original `egraph.analysis.pts` in the `enumerator` struct.

```rust
fn rebuild(&mut self, pts: &Vec<IOPair>) {
    // Update the stored pts
    self.egraph.analysis.pts = pts.clone();
    let egraph_clone = self.egraph.clone();

    let compute_data = |lc: &ArithLanguage| {
        ObsEquiv::make(&egraph_clone, lc)
    };
    let mut visited = HashSet::new();
    for id in self.id_order.iter() {
        let id = self.egraph.find(*id);
        if visited.contains(&id) {
            continue;
        } else {
        }
        visited.insert(id);
        // currently, our implementation should not consider the laziness, and egraph congurence should always be retained after the `split_eclass` method
        self.egraph.split_eclass(id, compute_data);
    }
}
```

In the `split_eclass` method, we should consider the following fields update when performing the split operation:

`EGraph`:

- `analysis`: $\checkmark$ in the `rebuild` method
- `unionfind`:
- `memo`:
- ~~`pending/analysis_pending`~~
- `classes`:
- ~~`classes_by_op`~~
- `clean`:

`EClass`:

- `id`: $\checkmark$ when initializing
- `nodes`: $\checkmark$ when initializing
- `data`:
- `parent`:

Psuedo code for the `split_eclass` method:

```rust
/// Splits an eclass into eclasses according to the result of compute_data function.
pub fn split_eclass(&mut self, id: Id, mut compute_data: impl FnMut(&L) -> N::Data)
    where N::Data: Eq + Hash + Clone + Debug
{
    let id: Id = self.find_mut(id);
    let clusters: HashMap<N::Data, Vec<usize>> = self.cluster_enodes_by_data(id, &mut compute_data);
    // Store the original eclass data for later use
    let original_eclass_parents: Vec<(L, Id)> = self.classes[&id].parents.clone();
    
    // Create new eclasses for each cluster and update data structures
    let mut new_eclasses: Vec<(Id, EClass<L, N::Data>)> = Vec::new();
    for (data, enode_indices) in clusters {
        let new_id: Id = self.unionfind.make_set();
        let new_class: EClass<L, N::Data> = EClass {
            id: new_id,
            nodes: enode_indices.iter().map(|&i| self.classes[&id].nodes[i].clone()).collect(),
            data,
            parents: Vec::new(),
        };

        new_eclasses.push((new_id, new_class));
    }

    // Update the memo before split the unionfind
    for (k, v) in self.memo.iter_mut() {
        
    }
    // TODO: Temporary workaround: we can never remove an id from the unionfind, so we route references to the original id to the first new eclass, hopefully this will not cause any issues
    // TODO: problem will occur if the memo is not updated for nodes in the new eclasses, as old id will be placed in the deprecated_leaders in the unionfind
    self.unionfind.split(id, )
    
    // update the memo, however, as we didn't introduce any new enodes, we don't need to update the memo except for the enode with the original id
    for (k, v) in memo {
        if v == id {
            // we need to find out the original enode from the nodes as 
        }
    }
    // TODO: Beaware that, all the ids in the new_eclasses are not in the memo, this may potentially cause some issues
    
}
```
