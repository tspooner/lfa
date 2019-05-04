use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Generics, Data, DataStruct, Field, Fields, Meta, Type, Ident};
use std::iter;

const WEIGHTS: &str = "weights";

struct Implementation {
    pub generics: Option<Vec<Type>>,
    pub body: Body,
}

impl Implementation {
    pub fn concrete(body: Body) -> Implementation {
        Implementation {
            generics: None,
            body,
        }
    }

    pub fn with_generics(generics: Vec<Type>, body: Body) -> Implementation {
        Implementation {
            generics: Some(generics),
            body,
        }
    }

    fn make_where_predicates(types: &Vec<Type>) -> Vec<syn::WherePredicate> {
        types.into_iter().map(|g| parse_quote! { #g: Parameterised }).collect()
    }

    pub fn add_trait_bounds(&self, mut generics: Generics) -> Generics {
        if let Some(ref gs) = self.generics {
            let new_ps = Self::make_where_predicates(gs);

            generics
                .make_where_clause()
                .predicates
                .extend(new_ps);
        }

        generics
    }
}

struct Body {
    pub weights: Option<TokenStream>,
    pub weights_dim: Option<TokenStream>,
    pub weights_view: TokenStream,
    pub weights_view_mut: TokenStream,
}

impl ToTokens for Body {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Some(ref weights_fn) = self.weights {
            tokens.extend(iter::once(quote! {
                fn weights(&self) -> Matrix<f64> { #weights_fn }
            }));
        }

        if let Some(ref weights_dim_fn) = self.weights_dim {
            tokens.extend(iter::once(quote! {
                fn weights_dim(&self) -> (usize, usize) { #weights_dim_fn }
            }));
        }

        let weights_view_fn = &self.weights_view;
        let weights_view_mut_fn = &self.weights_view_mut;

        tokens.extend(iter::once(quote! {
            fn weights_view(&self) -> MatrixView<f64> { #weights_view_fn }
        }).chain(iter::once(quote! {
            fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { #weights_view_mut_fn }
        })));
    }
}

struct WeightsField<'a, I: ToTokens> {
    pub accessor: I,
    pub field: &'a Field,
}

impl<'a, I: ToTokens> WeightsField<'a, I> {
    pub fn type_ident(&self) -> &'a Ident {
        match self.field.ty {
            Type::Path(ref tp) => {
                &tp.path.segments[0].ident
            },
            _ => unimplemented!(),
        }
    }
}

pub fn expand_derive_parameterised(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let implementation = parameterised_impl(&ast.data);

    let body = &implementation.body;
    let generics = implementation.add_trait_bounds(ast.generics.clone());
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    TokenStream::from(quote! {
        impl #impl_generics Parameterised for #name #ty_generics #where_clause { #body }
    })
}

fn parameterised_impl(data: &Data) -> Implementation {
    match data {
        Data::Struct(ref ds) => parameterised_struct_impl(ds),
        _ => unimplemented!(),
    }
}

fn parameterised_struct_impl(ds: &DataStruct) -> Implementation {
    let n_fields = ds.fields.iter().len();

    if n_fields > 1 {
        let mut annotated_fields: Vec<_> = ds.fields
            .iter()
            .enumerate()
            .filter(|(_, f)| has_weight_attribute(f))
            .map(|(i, f)| WeightsField {
                accessor: f.ident.clone().map(|i| quote! { #i }).unwrap_or(quote! { #i }),
                field: f,
            })
            .collect();

        if annotated_fields.is_empty() {
            let (index, iwf) = ds.fields.iter().enumerate().find(|(_, f)| match &f.ident {
                Some(ident) => ident.to_string() == WEIGHTS,
                None => false,
            }).expect("Couldn't infer weights field, consider annotating with #[weights].");

            parameterised_wf_impl(WeightsField {
                accessor: iwf.ident.clone().map(|i| quote! { #i }).unwrap_or(quote! { #index }),
                field: iwf,
            })
        } else if annotated_fields.len() == 1 {
            parameterised_wf_impl(annotated_fields.pop().unwrap())
        } else {
            panic!("Duplicate #[weights] annotations - \
                automatic view concatenation implementations are not currently supported.")
        }

    } else if n_fields == 1 {
        match ds.fields {
            Fields::Unnamed(ref fs) => parameterised_wf_impl(WeightsField {
                accessor: quote!{ 0 },
                field: &fs.unnamed[0],
            }),
            Fields::Named(ref fs) => parameterised_wf_impl(WeightsField {
                accessor: fs.named[0].ident.clone(),
                field: &fs.named[0],
            }),
            _ => unreachable!(),
        }
    } else {
        panic!("Nothing to derive Parameterised from!")
    }
}

fn parameterised_wf_impl<I: ToTokens>(wf: WeightsField<I>) -> Implementation {
    let accessor = &wf.accessor;
    let type_ident = wf.type_ident();

    match type_ident.to_string().as_ref() {
        "Vec" => Implementation::concrete(Body {
            weights: Some(quote! {
                let n_rows = self.#accessor.len();

                Matrix::from_shape_vec((n_rows, 1), self.#accessor.clone()).unwrap()
            }),
            weights_dim: Some(quote! { (self.#accessor.len(), 1) }),
            weights_view: quote! {
                let n_rows = self.#accessor.len();

                MatrixView::from_shape((n_rows, 1), &self.#accessor).unwrap()
            },
            weights_view_mut: quote! {
                let n_rows = self.#accessor.len();

                MatrixViewMut::from_shape((n_rows, 1), &mut self.#accessor).unwrap()
            },
        }),
        "Vector" => Implementation::concrete(Body {
            weights: None,
            weights_dim: Some(quote! { (self.#accessor.dim(), 1) }),
            weights_view: quote! {
                let n_rows = self.#accessor.len();

                self.#accessor.view().into_shape((n_rows, 1)).unwrap()
            },
            weights_view_mut: quote! {
                let n_rows = self.#accessor.len();

                self.#accessor.view_mut().into_shape((n_rows, 1)).unwrap()
            },
        }),
        "Matrix" => Implementation::concrete(Body {
            weights: Some(quote! { self.#accessor.clone() }),
            weights_dim: Some(quote! { self.#accessor.dim() }),
            weights_view: quote! { self.#accessor.view() },
            weights_view_mut: quote! { self.#accessor.view_mut() },
        }),
        _ => Implementation::with_generics(
            vec![wf.field.ty.clone()],
            Body {
                weights: Some(quote! { self.#accessor.weights() }),
                weights_dim: Some(quote! { self.#accessor.weights_dim() }),
                weights_view: quote! { self.#accessor.weights_view() },
                weights_view_mut: quote! { self.#accessor.weights_view_mut() },
            }
        )
    }
}

fn has_weight_attribute(f: &Field) -> bool {
    f.attrs.iter().any(|a| {
        a.parse_meta().map(|meta| {
            match meta {
                Meta::Word(ref ident) if ident.to_string() == WEIGHTS => true,
                _ => false,
            }
        }).unwrap_or(false)
    })
}
