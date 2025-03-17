pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "Vectors must be of the same length");

  let mut dot_product = 0.0;
  let mut norm_a = 0.0;
  let mut norm_b = 0.0;

  for i in 0..a.len() {
      dot_product += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
  }

  return dot_product / (norm_a.sqrt() * norm_b.sqrt());
}